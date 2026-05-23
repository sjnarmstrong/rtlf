use pyo3::prelude::*;

mod core;
mod error;
mod python;

#[pymodule]
fn rtlf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyRealtimeLazyFrame>()?;
    Ok(())
}
