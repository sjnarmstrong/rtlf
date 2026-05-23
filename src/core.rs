use std::collections::HashMap;
use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::datatypes::DataType;
use polars_core::schema::{Schema, SchemaExt};
use polars_core::prelude::Column;
use polars_expr::state::ExecutionState;
use polars_lazy::frame::{IntoLazy, LazyFrame};
use polars_mem_engine::create_physical_plan;
use polars_plan::plans::{ArenaLpIter, IR, IRPlan};
use polars_utils::arena::Node;


// We make use of DataFrame scan operations with specific names as placeholders
// we then substatute them out before creating the physical plan
const PLACEHOLDER_MARKER: &str = "__rtlf_placeholder__";

fn placeholder_name_from_ir(ir: &IR) -> Option<String> {
    if let IR::DataFrameScan { schema, .. } = ir {
        schema.iter_names().find_map(|name| {
            name.as_str().strip_prefix(PLACEHOLDER_MARKER).map(String::from)
        })
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
        let marker_name = format!("{}{}", PLACEHOLDER_MARKER, name);
        // Build columns: marker (Null typed) + user's columns
        let mut columns: Vec<Column> = Vec::with_capacity(schema.len() + 1);
        columns.push(Column::new_empty(marker_name.as_str().into(), &DataType::Null));
        for field in schema.iter_fields() {
            columns.push(Column::new_empty(field.name.clone(), &field.dtype));
        }
        let df = DataFrame::new(0, columns).expect("placeholder DataFrame construction failed");
        df.lazy()
    }
}
