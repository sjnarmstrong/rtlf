use std::collections::HashMap;
use std::sync::Arc;

use polars_buffer::Buffer;
use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::schema::{Schema, SchemaRef};
use polars_expr::state::ExecutionState;
use polars_lazy::frame::LazyFrame;
use polars_mem_engine::create_physical_plan;
use polars_plan::dsl::{DslPlan, FileScanDsl, ScanSources, UnifiedScanArgs};
use polars_plan::plans::{ArenaLpIter, IR, IRPlan};
use polars_utils::arena::Node;
use polars_utils::pl_path::PlRefPath;

const PLACEHOLDER_TOKEN: &str = "_rtlf::placeholder";

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
    pub ir_plan: IRPlan,
    pub placeholder_nodes: HashMap<String, usize>,
}

impl RealtimeLazyFrame {
    pub fn new(lf: LazyFrame) -> PolarsResult<Self> {
        let ir_plan = lf.to_alp_optimized()?;
        let mut placeholder_nodes = HashMap::new();
        for (node, ir) in ir_plan.lp_arena.iter(ir_plan.lp_top) {
            if let Some(name) = placeholder_name_from_ir(ir) {
                placeholder_nodes.insert(name, node.0);
            }
        }
        Ok(Self { ir_plan, placeholder_nodes })
    }

    pub fn collect(&self, inputs: HashMap<String, DataFrame>) -> PolarsResult<DataFrame> {
        let mut lp_arena = self.ir_plan.lp_arena.clone();
        let mut expr_arena = self.ir_plan.expr_arena.clone();
        let root = self.ir_plan.lp_top;

        for (name, &node_idx) in &self.placeholder_nodes {
            let df = inputs.get(name).ok_or_else(|| {
                polars_core::error::polars_err!(
                    ComputeError: "placeholder '{}' not provided in inputs; got: {:?}",
                    name,
                    inputs.keys().collect::<Vec<_>>()
                )
            })?;
            let schema = df.schema().clone();
            *lp_arena.get_mut(Node(node_idx)) = IR::DataFrameScan {
                df: Arc::new(df.clone()),
                schema,
                output_schema: None,
            };
        }

        let mut physical = create_physical_plan(root, &mut lp_arena, &mut expr_arena, None)?;
        physical.execute(&mut ExecutionState::new())
    }

    pub fn read_placeholder(name: &str, schema: &Schema) -> LazyFrame {
        let schema: SchemaRef = Arc::new(schema.clone());
        let sources = ScanSources::Paths(Buffer::from_iter([
            PlRefPath::new(PLACEHOLDER_TOKEN),
            PlRefPath::new(name),
        ]));
        let scan = DslPlan::Scan {
            sources,
            scan_type: Box::new(FileScanDsl::Parquet {
                options: polars_io::parquet::read::ParquetOptions {
                    schema: Some(schema),
                    ..Default::default()
                },
            }),
            unified_scan_args: Box::new(UnifiedScanArgs {
                glob: false,
                ..Default::default()
            }),
            cached_ir: Default::default(),
        };
        LazyFrame::from(scan)
    }
}
