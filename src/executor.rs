use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_expr::state::ExecutionState;
use polars_mem_engine::Executor;
use polars_plan::plans::IR;
use polars_utils::arena::{Arena, Node};

pub const PLACEHOLDER_TOKEN: &str = "_rtlf::placeholder";

/// A shared slot through which `collect()` injects a DataFrame directly into
/// the compiled executor — no clone, no ExecutionState cache.
pub type Slot = Arc<Mutex<Option<DataFrame>>>;

thread_local! {
    pub static PLACEHOLDER_REGISTRY: RefCell<Option<HashMap<String, Slot>>> =
        const { RefCell::new(None) };
}

pub struct PlaceholderExec {
    pub slot: Slot,
    /// Columns to select after retrieval — mirrors the projection pushdown that
    /// polars bakes into `IR::Scan { output_schema }` for real file scans.
    pub projection: Option<SchemaRef>,
}

impl Executor for PlaceholderExec {
    fn execute(&mut self, _state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let df = self.slot
            .lock()
            .expect("placeholder slot poisoned")
            .take()
            .ok_or_else(|| {
                polars_core::error::polars_err!(
                    ComputeError: "placeholder slot empty — collect() was not called before execute()"
                )
            })?;

        if let Some(schema) = &self.projection {
            let cols: Vec<_> = schema.iter_names().cloned().collect();
            Ok(df.select(cols)?)
        } else {
            Ok(df)
        }
    }
}

/// Bare function pointer matching `StreamingExecutorBuilder`. Intercepts
/// placeholder scan nodes during `create_physical_plan` and wires each one
/// to a freshly-created `Slot` that is registered in `PLACEHOLDER_REGISTRY`.
/// All other nodes (real scans, group_by, joins) are delegated to polars'
/// built-in streaming executor.
pub fn placeholder_builder(
    node: Node,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut polars_utils::arena::Arena<polars_plan::plans::AExpr>,
) -> PolarsResult<Box<dyn Executor>> {
    let ir = lp_arena.get(node);
    let name = placeholder_name_from_ir(ir);
    // Extract projection before the borrow ends.
    let projection: Option<SchemaRef> = if name.is_some() {
        if let IR::Scan { output_schema, .. } = ir {
            output_schema.clone()
        } else {
            None
        }
    } else {
        None
    };

    if let Some(name) = name {
        let slot: Slot = Arc::new(Mutex::new(None));
        PLACEHOLDER_REGISTRY.with(|r| {
            r.borrow_mut()
                .as_mut()
                .expect("placeholder_builder called outside of CompiledRealtimeLazyFrame::from_parts")
                .insert(name, slot.clone());
        });
        return Ok(Box::new(PlaceholderExec { slot, projection }));
    }
    polars_stream::build_streaming_query_executor(node, lp_arena, expr_arena)
}

pub fn placeholder_name_from_ir(ir: &IR) -> Option<String> {
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
