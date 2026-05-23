use polars_core::error::PolarsError;
use pyo3::PyErr;

/// Newtype so we can impl From<PolarsError> for PyErr (orphan rule).
pub struct PyRtlfErr(pub PolarsError);

impl From<PolarsError> for PyRtlfErr {
    fn from(e: PolarsError) -> Self {
        Self(e)
    }
}

impl From<PyRtlfErr> for PyErr {
    fn from(e: PyRtlfErr) -> PyErr {
        use PolarsError::*;
        match e.0 {
            ColumnNotFound(msg) => pyo3::exceptions::PyKeyError::new_err(msg.to_string()),
            ComputeError(msg) => pyo3::exceptions::PyRuntimeError::new_err(msg.to_string()),
            IO { error, .. } => pyo3::exceptions::PyIOError::new_err(error.to_string()),
            other => pyo3::exceptions::PyRuntimeError::new_err(other.to_string()),
        }
    }
}
