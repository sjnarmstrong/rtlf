use pyo3::prelude::*;

mod compiled;
mod error;
mod executor;
mod python;
mod realtime;

#[pymodule]
fn rtlf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyRealtimeLazyFrame>()?;
    m.add_class::<python::PyCompiledRealtimeLazyFrame>()?;
    Ok(())
}
