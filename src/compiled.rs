use std::collections::HashMap;
use std::sync::Mutex;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_expr::state::ExecutionState;
use polars_mem_engine::{Executor, StreamingExecutorBuilder, create_physical_plan};
use polars_plan::plans::{AExpr, ArenaLpIter as _, IR};
use polars_utils::arena::{Arena, Node};

use crate::executor::{Slot, placeholder_builder, placeholder_name_from_ir, PLACEHOLDER_REGISTRY};

/// A physical plan compiled once from a `RealtimeLazyFrame`. Each `collect()`
/// call injects DataFrames directly into executor slots — no plan recompilation,
/// no clone, no `ExecutionState` cache overhead.
///
/// Limitation: queries whose streaming executors carry state between runs
/// (e.g. `concat`/`union`) cannot be re-executed; use `RealtimeLazyFrame` for
/// those.
pub struct CompiledRealtimeLazyFrame {
    /// Mutex because `Executor::execute` takes `&mut self`. Concurrent callers
    /// are serialised; for true parallelism create one instance per thread.
    physical_plan: Mutex<Box<dyn Executor>>,
    placeholder_slots: HashMap<String, Slot>,
}

unsafe impl Send for CompiledRealtimeLazyFrame {}
unsafe impl Sync for CompiledRealtimeLazyFrame {}

impl CompiledRealtimeLazyFrame {
    /// Called by `RealtimeLazyFrame::compile()` with cloned arenas.
    pub(crate) fn from_parts(
        lp_top: Node,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<Self> {
        // Lift pushed-down predicates out of placeholder scans into explicit
        // Filter nodes so the compiled executor tree applies them correctly.
        let nodes_with_predicates: Vec<_> = lp_arena
            .iter(lp_top)
            .filter_map(|(node, ir)| {
                if placeholder_name_from_ir(ir).is_none() {
                    return None;
                }
                if let IR::Scan { predicate: Some(pred), .. } = ir {
                    Some((node, pred.clone()))
                } else {
                    None
                }
            })
            .collect();

        for (node, pred) in nodes_with_predicates {
            if let IR::Scan { predicate, .. } = lp_arena.get_mut(node) {
                *predicate = None;
            }
            let clean_scan = lp_arena.get(node).clone();
            let scan_node = lp_arena.add(clean_scan);
            *lp_arena.get_mut(node) = IR::Filter { input: scan_node, predicate: pred };
        }

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

        let mut plan = self.physical_plan.lock().expect("executor mutex poisoned");

        for (name, slot) in &self.placeholder_slots {
            *slot.lock().expect("placeholder slot poisoned") = inputs.remove(name);
        }

        plan.execute(&mut ExecutionState::new())
    }
}
