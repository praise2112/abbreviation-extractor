use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};

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
