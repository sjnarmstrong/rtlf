use std::collections::HashMap;

use polars_core::datatypes::Field;
use polars_core::frame::DataFrame;
use polars_core::schema::Schema;
use pyo3::pybacked::PyBackedStr;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_polars::{PyDataFrame, PyDataType, PyLazyFrame};

use crate::compiled::CompiledRealtimeLazyFrame;
use crate::error::PyRtlfErr;
use crate::realtime::RealtimeLazyFrame;

fn extract_schema(ob: &Bound<'_, PyAny>) -> PyResult<Schema> {
    let dict = ob.cast::<PyDict>()?;
    dict.iter()
        .map(|(k, v)| {
            let name = k.extract::<PyBackedStr>()?;
            let dtype = v.extract::<PyDataType>()?.0;
            Ok(Field::new((&*name).into(), dtype))
        })
        .collect::<PyResult<Schema>>()
}

fn unwrap_inputs(inputs: HashMap<String, PyDataFrame>) -> HashMap<String, DataFrame> {
    inputs.into_iter().map(|(k, v)| (k, v.0)).collect()
}

// ── RealtimeLazyFrame ────────────────────────────────────────────────────────

#[pyclass]
pub struct PyRealtimeLazyFrame {
    inner: RealtimeLazyFrame,
}

#[pymethods]
impl PyRealtimeLazyFrame {
    #[new]
    fn new(lf: PyLazyFrame) -> PyResult<Self> {
        let inner = RealtimeLazyFrame::new(lf.0).map_err(PyRtlfErr::from)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn read_placeholder(name: String, schema: &Bound<'_, PyAny>) -> PyResult<PyLazyFrame> {
        let schema = extract_schema(schema)?;
        Ok(PyLazyFrame(RealtimeLazyFrame::read_placeholder(&name, &schema)))
    }

    /// Compile the stored IR plan into a `CompiledRealtimeLazyFrame`.
    /// The compilation clones the arenas so this instance remains usable.
    fn compile(&self) -> PyResult<PyCompiledRealtimeLazyFrame> {
        self.inner
            .compile()
            .map(|inner| PyCompiledRealtimeLazyFrame { inner })
            .map_err(|e| pyo3::PyErr::from(PyRtlfErr::from(e)))
    }

    /// Collect by re-compiling the physical plan on every call.
    fn collect(&self, py: Python<'_>, inputs: HashMap<String, PyDataFrame>) -> PyResult<PyDataFrame> {
        let rust_inputs = unwrap_inputs(inputs);
        py.detach(|| {
            self.inner
                .collect(rust_inputs)
                .map(PyDataFrame)
                .map_err(|e| pyo3::PyErr::from(PyRtlfErr::from(e)))
        })
    }
}

// ── CompiledRealtimeLazyFrame ────────────────────────────────────────────────

#[pyclass]
pub struct PyCompiledRealtimeLazyFrame {
    inner: CompiledRealtimeLazyFrame,
}

#[pymethods]
impl PyCompiledRealtimeLazyFrame {
    /// Collect using pre-compiled physical plan with zero-copy slot injection.
    fn collect(&self, py: Python<'_>, inputs: HashMap<String, PyDataFrame>) -> PyResult<PyDataFrame> {
        let rust_inputs = unwrap_inputs(inputs);
        py.detach(|| {
            self.inner
                .collect(rust_inputs)
                .map(PyDataFrame)
                .map_err(|e| pyo3::PyErr::from(PyRtlfErr::from(e)))
        })
    }
}
