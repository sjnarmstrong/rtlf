use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use polars_buffer::Buffer;
use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::schema::{Schema, SchemaRef};
use polars_expr::state::ExecutionState;
use polars_lazy::frame::LazyFrame;
use polars_mem_engine::create_physical_plan;
use polars_plan::dsl::{DslPlan, FileScanDsl, FileScanIR, ScanSources, UnifiedScanArgs};
use polars_plan::plans::{AExpr, ArenaLpIter as _, FileInfo, IR};
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_path::PlRefPath;

use crate::compiled::CompiledRealtimeLazyFrame;
use crate::executor::placeholder_name_from_ir;
use crate::executor::PLACEHOLDER_TOKEN;

/// Stores an optimised IR plan and re-runs `create_physical_plan` on every
/// `collect()`.  The per-call compilation cost is constant in expression
/// complexity, so this is appropriate for one-shot or low-frequency use.
/// For high-frequency hot-path use, call `compile()` to get a
/// `CompiledRealtimeLazyFrame` that compiles the plan once and then executes
/// with direct slot injection on every call.
pub struct RealtimeLazyFrame {
    lp_top: Node,
    lp_arena: Arena<IR>,
    expr_arena: Arena<AExpr>,
    /// Node index in `lp_arena` for each placeholder scan, keyed by name.
    placeholder_nodes: HashMap<String, Node>,
}

// Arena<IR/AExpr> are Send+Sync when IR/AExpr are Send+Sync.
unsafe impl Send for RealtimeLazyFrame {}
unsafe impl Sync for RealtimeLazyFrame {}

impl RealtimeLazyFrame {
    pub fn new(lf: LazyFrame) -> PolarsResult<Self> {
        let ir_plan = lf.to_alp_optimized()?;
        let lp_top = ir_plan.lp_top;
        let lp_arena = ir_plan.lp_arena;
        let expr_arena = ir_plan.expr_arena;

        let mut placeholder_nodes = HashMap::new();
        for (node, ir) in lp_arena.iter(lp_top) {
            if let Some(name) = placeholder_name_from_ir(ir) {
                placeholder_nodes.insert(name, node);
            }
        }

        Ok(Self {
            lp_top,
            lp_arena,
            expr_arena,
            placeholder_nodes,
        })
    }

    /// Compile the plan once into a `CompiledRealtimeLazyFrame`. The
    /// compilation clones the arenas so `self` remains usable afterwards.
    pub fn compile(&self) -> PolarsResult<CompiledRealtimeLazyFrame> {
        let mut lp_arena = self.lp_arena.clone();
        let mut expr_arena = self.expr_arena.clone();
        CompiledRealtimeLazyFrame::from_parts(self.lp_top, &mut lp_arena, &mut expr_arena)
    }

    /// Re-compiles the physical plan from the stored IR on every call.
    /// Inputs are injected by replacing placeholder `IR::Scan` nodes with
    /// `IR::DataFrameScan` nodes in a cloned arena.
    pub fn collect(&self, mut inputs: HashMap<String, DataFrame>) -> PolarsResult<DataFrame> {
        for name in self.placeholder_nodes.keys() {
            if !inputs.contains_key(name) {
                polars_core::error::polars_bail!(
                    ComputeError: "placeholder '{}' not provided; got: {:?}",
                    name,
                    inputs.keys().collect::<Vec<_>>()
                );
            }
        }

        let mut lp_arena = self.lp_arena.clone();
        let mut expr_arena = self.expr_arena.clone();

        for (name, &node) in &self.placeholder_nodes {
            // Extract schema before taking a mutable borrow.
            let schema: SchemaRef = match lp_arena.get(node) {
                IR::Scan { file_info, .. } => file_info.schema.clone(),
                _ => unreachable!("placeholder node was not IR::Scan"),
            };
            *lp_arena.get_mut(node) = IR::DataFrameScan {
                df: Arc::new(inputs.remove(name).expect("validated above")),
                schema,
                output_schema: None,
            };
        }

        let mut physical_plan =
            create_physical_plan(self.lp_top, &mut lp_arena, &mut expr_arena, None)?;
        physical_plan.execute(&mut ExecutionState::new())
    }

    /// Build a placeholder `LazyFrame` for use inside the query passed to
    /// `new()`.  Setting `ParquetOptions::schema` bypasses the filesystem
    /// metadata fetch during `to_alp_optimized()` and, unlike `cached_ir`,
    /// survives the Python serialisation round-trip.
    pub fn read_placeholder(name: &str, schema: &Schema) -> LazyFrame {
        let schema_ref: SchemaRef = Arc::new(schema.clone());

        let sources = ScanSources::Paths(Buffer::from_iter([
            PlRefPath::new(PLACEHOLDER_TOKEN),
            PlRefPath::new(name),
        ]));

        let file_info = FileInfo {
            schema: schema_ref.clone(),
            reader_schema: None,
            row_estimation: (None, usize::MAX),
        };
        let ir_scan = IR::Scan {
            sources: sources.clone(),
            file_info,
            hive_parts: None,
            predicate: None,
            predicate_file_skip_applied: None,
            output_schema: None,
            scan_type: Box::new(FileScanIR::Parquet {
                options: polars_io::parquet::read::ParquetOptions::default(),
                metadata: None,
            }),
            unified_scan_args: Box::new(UnifiedScanArgs {
                glob: false,
                ..Default::default()
            }),
        };

        let scan = DslPlan::Scan {
            sources,
            scan_type: Box::new(FileScanDsl::Parquet {
                options: polars_io::parquet::read::ParquetOptions {
                    schema: Some(schema_ref.clone()),
                    ..Default::default()
                },
            }),
            unified_scan_args: Box::new(UnifiedScanArgs {
                glob: false,
                ..Default::default()
            }),
            cached_ir: Arc::new(Mutex::new(Some(ir_scan))),
        };

        LazyFrame::from(scan)
    }
}
