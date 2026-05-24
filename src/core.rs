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

const PLACEHOLDER_TOKEN: &str = "_rtlf::placeholder";

// Each placeholder owns an Arc<Mutex<Option<DataFrame>>> that is shared between
// the PlaceholderExec (which reads it) and RealtimeLazyFrame (which writes it).
// DataFrame moves in on collect(), moves out on execute() — no clone anywhere.
type Slot = Arc<Mutex<Option<DataFrame>>>;

thread_local! {
    static PLACEHOLDER_REGISTRY: RefCell<Option<HashMap<String, Slot>>> =
        const { RefCell::new(None) };
}

struct PlaceholderExec {
    slot: Slot,
}

impl Executor for PlaceholderExec {
    fn execute(&mut self, _state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        self.slot
            .lock()
            .expect("placeholder slot poisoned")
            .take()
            .ok_or_else(|| {
                polars_core::error::polars_err!(
                    ComputeError: "placeholder slot empty — collect() was not called before execute()"
                )
            })
    }
}

fn placeholder_builder(
    node: Node,
    lp_arena: &mut Arena<IR>,
    _expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<Box<dyn Executor>> {
    let ir = lp_arena.get(node);
    if let Some(name) = placeholder_name_from_ir(ir) {
        let slot: Slot = Arc::new(Mutex::new(None));
        PLACEHOLDER_REGISTRY.with(|r| {
            r.borrow_mut()
                .as_mut()
                .expect("placeholder_builder called outside of RealtimeLazyFrame::new")
                .insert(name, slot.clone());
        });
        return Ok(Box::new(PlaceholderExec { slot }));
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
    // Mutex because Executor::execute takes &mut self. Concurrent callers are
    // serialised; for true parallelism create one RealtimeLazyFrame per thread.
    physical_plan: Mutex<Box<dyn Executor>>,
    placeholder_slots: HashMap<String, Slot>,
}

// Slot is Arc<Mutex<...>> which is Send+Sync; DataFrame is Send.
unsafe impl Send for RealtimeLazyFrame {}
unsafe impl Sync for RealtimeLazyFrame {}

impl RealtimeLazyFrame {
    pub fn new(lf: LazyFrame) -> PolarsResult<Self> {
        let mut ir_plan = lf.to_alp_optimized()?;

        PLACEHOLDER_REGISTRY.with(|r| *r.borrow_mut() = Some(HashMap::new()));

        let physical_plan = create_physical_plan(
            ir_plan.lp_top,
            &mut ir_plan.lp_arena,
            &mut ir_plan.expr_arena,
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

        // Lock the plan first so concurrent collect() calls are serialised while
        // we fill the slots. The slots are only valid between the fill and execute.
        let mut plan = self.physical_plan.lock().expect("executor mutex poisoned");

        for (name, slot) in &self.placeholder_slots {
            // remove() moves the DataFrame into the slot — no clone, no Arc increment.
            *slot.lock().expect("placeholder slot poisoned") = inputs.remove(name);
        }

        plan.execute(&mut ExecutionState::new())
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
