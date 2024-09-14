use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::error::Error;

/// Represents an abbreviation-definition pair with its position in the text.
#[pyclass]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbbreviationDefinition {
    #[pyo3(get, set)]
    pub abbreviation: String,
    #[pyo3(get, set)]
    pub definition: String,
    #[pyo3(get, set)]
    pub start: usize,
    #[pyo3(get, set)]
    pub end: usize,
}

#[pymethods]
impl AbbreviationDefinition {
    /// Creates a new AbbreviationDefinition instance.
    #[new]
    pub fn new(abbreviation: String, definition: String, start: usize, end: usize) -> Self {
        AbbreviationDefinition {
            abbreviation,
            definition,
            start,
            end,
        }
    }

    /// Returns a string representation of the AbbreviationDefinition.
    fn __repr__(&self) -> String {
        format!(
            "AbbreviationDefinition(abbreviation='{}', definition='{}', start={}, end={})",
            self.abbreviation, self.definition, self.start, self.end
        )
    }

    /// Deserializes the object from a byte string (used for Python pickling).
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }

    /// Serializes the object to a byte string (used for Python pickling).
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new_bound(py, &serialize(&self).unwrap()))
    }

    /// Returns the arguments needed to recreate this object (used for Python pickling).
    fn __getnewargs__(&self) -> PyResult<(String, String, usize, usize)> {
        Ok((
            self.abbreviation.clone(),
            self.definition.clone(),
            self.start,
            self.end,
        ))
    }
}

#[derive(Clone, Copy)]
pub struct AbbreviationOptions {
    pub most_common_definition: bool,
    pub first_definition: bool,
    pub tokenize: bool,
}

impl Default for AbbreviationOptions {
    fn default() -> Self {
        Self {
            most_common_definition: false,
            first_definition: false,
            tokenize: true,
        }
    }
}

impl AbbreviationOptions {
    pub fn new(most_common_definition: bool, first_definition: bool, tokenize: bool) -> Self {
        Self {
            most_common_definition,
            first_definition,
            tokenize,
        }
    }
}

#[derive(Clone, Copy)]
pub struct FileExtractionOptions {
    pub num_threads: usize,
    pub chunk_size: usize,
    pub show_progress: bool,
}

impl Default for FileExtractionOptions {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            chunk_size: 1024 * 1024, // 1 MB chunks
            show_progress: true,
        }
    }
}

impl FileExtractionOptions {
    pub fn new(num_threads: usize, chunk_size: usize, show_progress: bool) -> Self {
        Self {
            num_threads,
            chunk_size,
            show_progress,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExtractionError {
    ProcessingError(String),
    IOError(String),
    ThreadPoolError(String),
}

impl fmt::Display for ExtractionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ExtractionError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
            ExtractionError::IOError(msg) => write!(f, "IO error: {}", msg),
            ExtractionError::ThreadPoolError(msg) => write!(f, "Thread pool error: {}", msg),
        }
    }
}

impl Error for ExtractionError {}

#[pyclass]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractionResult {
    #[pyo3(get, set)]
    pub extractions: Vec<AbbreviationDefinition>,
    #[pyo3(get, set)]
    pub errors: Vec<ExtractionError>,
}

#[pymethods]
impl ExtractionResult {
    #[new]
    pub fn new(extractions: Vec<AbbreviationDefinition>, errors: Vec<ExtractionError>) -> Self {
        ExtractionResult {
            extractions,
            errors,
        }
    }

    fn __repr__(&self) -> String {
        let extractions_repr = if self.extractions.is_empty() {
            "[]".to_string()
        } else {
            let samples = self
                .extractions
                .iter()
                .take(5)
                .map(|ad| {
                    format!(
                        "AbbreviationDefinition(abbreviation='{}', definition='{}', ...)",
                        ad.abbreviation, ad.definition
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");

            if self.extractions.len() > 5 {
                format!("[{}, ... and {} more]", samples, self.extractions.len() - 5)
            } else {
                format!("[{}]", samples)
            }
        };

        let errors_repr = if self.errors.is_empty() {
            "[]".to_string()
        } else {
            let samples = self
                .errors
                .iter()
                .take(5)
                .map(|e| match e {
                    ExtractionError::ProcessingError(msg) => format!("ProcessingError('{}')", msg),
                    ExtractionError::IOError(msg) => format!("IOError('{}')", msg),
                    ExtractionError::ThreadPoolError(msg) => format!("ThreadPoolError('{}')", msg),
                })
                .collect::<Vec<_>>()
                .join(", ");

            if self.errors.len() > 5 {
                format!("[{}, ... and {} more]", samples, self.errors.len() - 5)
            } else {
                format!("[{}]", samples)
            }
        };

        format!(
            "ExtractionResult(extractions={}, errors={})",
            extractions_repr, errors_repr
        )
    }

    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new_bound(py, &serialize(&self).unwrap()))
    }

    fn __getnewargs__(&self) -> PyResult<(Vec<AbbreviationDefinition>, Vec<ExtractionError>)> {
        Ok((self.extractions.clone(), self.errors.clone()))
    }
}
