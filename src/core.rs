use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use polars_buffer::Buffer;
use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::schema::{Schema, SchemaRef};
use polars_expr::state::ExecutionState;
use polars_lazy::frame::LazyFrame;
use polars_mem_engine::{Executor, StreamingExecutorBuilder, create_physical_plan};
use polars_plan::dsl::{DslPlan, FileScanDsl, FileScanIR, ScanSources, UnifiedScanArgs};
use polars_plan::plans::{AExpr, FileInfo, IR};
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_path::PlRefPath;
use polars_utils::unique_id::UniqueId;

const PLACEHOLDER_TOKEN: &str = "_rtlf::placeholder";

// ── Thread-local: populated by placeholder_builder during create_physical_plan, ──
// ── harvested by new() immediately after.                                        ──
thread_local! {
    static PLACEHOLDER_REGISTRY: RefCell<Option<HashMap<String, UniqueId>>> =
        const { RefCell::new(None) };
}

// ── Executor that reads a pre-populated DataFrame from the ExecutionState cache ──
struct PlaceholderExec {
    id: UniqueId,
}

impl Executor for PlaceholderExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        Ok(state.get_df_cache(&self.id))
    }
}

// ── StreamingExecutorBuilder (bare fn pointer) that intercepts placeholder scans ──
fn placeholder_builder(
    node: Node,
    lp_arena: &mut Arena<IR>,
    _expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<Box<dyn Executor>> {
    let ir = lp_arena.get(node);
    if let Some(name) = placeholder_name_from_ir(ir) {
        let id = UniqueId::new();
        PLACEHOLDER_REGISTRY.with(|r| {
            r.borrow_mut()
                .as_mut()
                .expect("placeholder_builder called outside of RealtimeLazyFrame::new")
                .insert(name, id);
        });
        return Ok(Box::new(PlaceholderExec { id }));
    }
    polars_core::error::polars_bail!(
        ComputeError: "non-placeholder file scan in RealtimeLazyFrame; \
                       all sources must come from read_placeholder()"
    )
}

fn placeholder_name_from_ir(ir: &IR) -> Option<String> {
    if let IR::Scan { sources, .. } = ir {
        let paths = sources.as_paths()?;
        if paths.len() != 2 || paths[0].as_str() != PLACEHOLDER_TOKEN {
            return None;
        }
        Some(paths[1].as_str().to_owned())
    } else {
        None
    }
}

pub struct RealtimeLazyFrame {
    // Compiled once in new(). Mutex because Executor::execute takes &mut self.
    // Concurrent callers will be serialised; for true parallelism, create
    // one RealtimeLazyFrame per thread.
    physical_plan: Mutex<Box<dyn Executor>>,
    placeholders: HashMap<String, UniqueId>,
}

// Arena<IR/AExpr> are Send; UniqueId and DataFrame are Send.
unsafe impl Send for RealtimeLazyFrame {}
unsafe impl Sync for RealtimeLazyFrame {}

impl RealtimeLazyFrame {
    pub fn new(lf: LazyFrame) -> PolarsResult<Self> {
        let mut ir_plan = lf.to_alp_optimized()?;

        // Activate registry so placeholder_builder can write into it.
        PLACEHOLDER_REGISTRY.with(|r| *r.borrow_mut() = Some(HashMap::new()));

        let physical_plan = create_physical_plan(
            ir_plan.lp_top,
            &mut ir_plan.lp_arena,
            &mut ir_plan.expr_arena,
            Some(placeholder_builder as StreamingExecutorBuilder),
        )?;

        let placeholders = PLACEHOLDER_REGISTRY
            .with(|r| r.borrow_mut().take())
            .unwrap_or_default();

        Ok(Self {
            physical_plan: Mutex::new(physical_plan),
            placeholders,
        })
    }

    pub fn collect(&self, inputs: HashMap<String, DataFrame>) -> PolarsResult<DataFrame> {
        for name in self.placeholders.keys() {
            if !inputs.contains_key(name) {
                polars_core::error::polars_bail!(
                    ComputeError: "placeholder '{}' not provided; got: {:?}",
                    name,
                    inputs.keys().collect::<Vec<_>>()
                );
            }
        }

        let mut state = ExecutionState::new();
        for (name, id) in &self.placeholders {
            // hit_count = 1: each placeholder is used exactly once per collect.
            state.set_df_cache(id, inputs[name].clone(), 1);
        }

        self.physical_plan
            .lock()
            .expect("executor mutex poisoned")
            .execute(&mut state)
    }

    /// Build a placeholder LazyFrame. Use inside the query passed to `new()`.
    ///
    /// The DSL scan node's `cached_ir` is pre-populated so polars never
    /// tries to open the fake path during `to_alp_optimized()`.
    pub fn read_placeholder(name: &str, schema: &Schema) -> LazyFrame {
        let schema_ref: SchemaRef = Arc::new(schema.clone());

        let sources = ScanSources::Paths(Buffer::from_iter([
            PlRefPath::new(PLACEHOLDER_TOKEN),
            PlRefPath::new(name),
        ]));

        // Build the IR::Scan directly so the DSL→IR conversion never opens the path.
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
                // Setting schema here bypasses filesystem metadata fetch during to_alp_optimized(),
                // and unlike cached_ir it survives Python serialization round-trip.
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
