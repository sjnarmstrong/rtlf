use std::collections::HashMap;
use std::sync::Mutex;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_expr::state::ExecutionState;
use polars_mem_engine::{Executor, StreamingExecutorBuilder, create_physical_plan};
use polars_plan::plans::AExpr;
use polars_plan::plans::IR;
use polars_utils::arena::{Arena, Node};

use crate::executor::{Slot, placeholder_builder, PLACEHOLDER_REGISTRY};

/// A physical plan compiled once from a `RealtimeLazyFrame`. Each `collect()`
/// call injects DataFrames directly into executor slots — no plan recompilation,
/// no clone, no `ExecutionState` cache overhead.
pub struct CompiledRealtimeLazyFrame {
    /// Mutex because `Executor::execute` takes `&mut self`. Concurrent callers
    /// are serialised; for true parallelism create one instance per thread.
    physical_plan: Mutex<Box<dyn Executor>>,
    placeholder_slots: HashMap<String, Slot>,
}

// Slot is Arc<Mutex<...>> which is Send+Sync; DataFrame and Executor are Send.
unsafe impl Send for CompiledRealtimeLazyFrame {}
unsafe impl Sync for CompiledRealtimeLazyFrame {}

impl CompiledRealtimeLazyFrame {
    /// Called by `RealtimeLazyFrame::compile()` with cloned arenas.
    pub(crate) fn from_parts(
        lp_top: Node,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<Self> {
        PLACEHOLDER_REGISTRY.with(|r| *r.borrow_mut() = Some(HashMap::new()));

        let physical_plan = create_physical_plan(
            lp_top,
            lp_arena,
            expr_arena,
            Some(placeholder_builder as StreamingExecutorBuilder),
        )?;

        let placeholder_slots = PLACEHOLDER_REGISTRY
            .with(|r| r.borrow_mut().take())
            .unwrap_or_default();

        Ok(Self {
            physical_plan: Mutex::new(physical_plan),
            placeholder_slots,
        })
    }

    pub fn collect(&self, mut inputs: HashMap<String, DataFrame>) -> PolarsResult<DataFrame> {
        for name in self.placeholder_slots.keys() {
            if !inputs.contains_key(name) {
                polars_core::error::polars_bail!(
                    ComputeError: "placeholder '{}' not provided; got: {:?}",
                    name,
                    inputs.keys().collect::<Vec<_>>()
                );
            }
        }

        // Lock the plan first so the slot-fill and execute are atomic with
        // respect to other concurrent collect() callers on the same instance.
        let mut plan = self.physical_plan.lock().expect("executor mutex poisoned");

        for (name, slot) in &self.placeholder_slots {
            // remove() moves the DataFrame into the slot — no Arc increment.
            *slot.lock().expect("placeholder slot poisoned") = inputs.remove(name);
        }

        plan.execute(&mut ExecutionState::new())
    }
}
