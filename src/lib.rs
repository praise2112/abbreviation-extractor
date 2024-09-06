//! # Abbreviation Extractor
//!
//! Abbreviation Extractor is a high-performance Rust library with Python bindings for extracting
//! abbreviation-definition pairs from text, particularly focused on biomedical literature. It implements
//! an improved version of the Schwartz-Hearst algorithm as described in:
//!
//! A. Schwartz and M. Hearst (2003) A Simple Algorithm for Identifying Abbreviations Definitions
//! in Biomedical Text. Biocomputing, 451-462.
//!
//! ## Overview
//!
//! This library provides functionality to extract abbreviation-definition pairs from text. It supports
//! both single-threaded and parallel processing, making it suitable for various text processing tasks.
//! The library is designed with a focus on biomedical literature but can be applied to other domains as well.
//!
//! Key components of the library include:
//! - Support for parallel processing of large datasets
//! - Customizable extraction parameters like selecting the most common or first definition for each abbreviation
//! - Python bindings for easy integration with Python projects
//! - Tokenization of input text for more accurate extraction
//!
//!
//! ## Usage
//!
//! ### Rust
//!
//! ```rust
//! use abbreviation_extractor::{extract_abbreviation_definition_pairs, AbbreviationOptions};
//!
//! let text = "The World Health Organization (WHO) is a specialized agency.";
//! let options = AbbreviationOptions::default();
//! let result = extract_abbreviation_definition_pairs(text, options);
//!
//! for pair in result {
//!     println!("Abbreviation: {}, Definition: {}", pair.abbreviation, pair.definition);
//! }
//! ```
//!
//! ### Python
//!
//! ```python
//! from abbreviation_extractor import extract_abbreviation_definition_pairs
//!
//! text = "The World Health Organization (WHO) is a specialized agency."
//! result = extract_abbreviation_definition_pairs(text)
//!
//! for pair in result:
//!     print(f"Abbreviation: {pair.abbreviation}, Definition: {pair.definition}")
//! ```
//!
//! ## Customizing Extraction
//!
//! You can customize the extraction process using `AbbreviationOptions`:
//!
//! ```rust
//! use abbreviation_extractor::{extract_abbreviation_definition_pairs, AbbreviationOptions};
//!
//! let text = "The World Health Organization (WHO) is a specialized agency. \
//!             The World Heritage Organization (WHO) is different.";
//!
//! // Get only the most common definition for each abbreviation
//! let options = AbbreviationOptions::new(true, false, true);
//! let result = extract_abbreviation_definition_pairs(text, options);
//!
//! // Get only the first definition for each abbreviation
//! let options = AbbreviationOptions::new(false, true, true);
//! let result = extract_abbreviation_definition_pairs(text, options);
//!
//! // Disable tokenization (if the input is already tokenized)
//! let options = AbbreviationOptions::new(false, false, false);
//! let result = extract_abbreviation_definition_pairs(text, options);
//! ```
//!
//! ## Functions
//!
//! The main functions provided by this library are:
//!
//! - [`extract_abbreviation_definition_pairs`]: Extracts abbreviation-definition pairs from a single text.
//! - [`extract_abbreviation_definition_pairs_parallel`]: Extracts abbreviation-definition pairs from multiple texts in parallel.
//!
//! For detailed information on each function, please refer to their individual documentation.
//!
//! ## Modules
//!
//! - [`candidate`]: Defines the `Candidate` struct used in the extraction process
//! - [`extraction`]: Contains the core logic for extracting abbreviation-definition pairs
//! - [`utils`]: Utility functions and regular expressions used in the extraction process
//! - [`abbreviation_definitions`]: Defines the `AbbreviationDefinition` and `AbbreviationOptions` structs

use pyo3::prelude::*;

pub mod abbreviation_definitions;
pub mod candidate;
pub mod extraction;
pub mod utils;

pub use abbreviation_definitions::{AbbreviationDefinition, AbbreviationOptions};
pub use candidate::Candidate;
pub use extraction::{
    best_candidates, extract_abbreviation_definition_pairs,
    extract_abbreviation_definition_pairs_parallel, get_definition, select_definition,
};

/// Extracts abbreviation-definition pairs from a single text.
///
/// This function is exposed to Python and serves as a wrapper around the Rust
/// `extract_abbreviation_definition_pairs` function.
///
/// # Arguments
///
/// * `text` - The input text to extract abbreviation-definition pairs from.
/// * `most_common_definition` - If `Some(true)`, only the most common definition for each
///   abbreviation is returned. Default is `None` (false).
/// * `first_definition` - If `Some(true)`, only the first definition for each abbreviation
///   is returned. Default is `None` (false).
/// * `tokenize` - If `Some(true)`, the input text is tokenized before processing. Default is `None` (true).
///
/// # Returns
///
/// A `PyResult` containing a vector of `AbbreviationDefinition` structs.
#[pyfunction]
#[pyo3(name = "extract_abbreviation_definition_pairs")]
#[pyo3(signature = (text, most_common_definition=None, first_definition=None, tokenize=None))]
fn py_extract_abbreviation_definition_pairs(
    text: &str,
    most_common_definition: Option<bool>,
    first_definition: Option<bool>,
    tokenize: Option<bool>,
) -> PyResult<Vec<AbbreviationDefinition>> {
    let options = AbbreviationOptions::new(
        most_common_definition.unwrap_or(false),
        first_definition.unwrap_or(false),
        tokenize.unwrap_or(true),
    );
    Ok(extract_abbreviation_definition_pairs(text, options))
}

/// Extracts abbreviation-definition pairs from multiple texts in parallel.
///
/// This function is exposed to Python and serves as a wrapper around the Rust
/// `extract_abbreviation_definition_pairs_parallel` function.
///
/// # Arguments
///
/// * `texts` - A vector of input texts to extract abbreviation-definition pairs from.
/// * `most_common_definition` - If `Some(true)`, only the most common definition for each
///   abbreviation is returned. Default is `None` (false).
/// * `first_definition` - If `Some(true)`, only the first definition for each abbreviation
///   is returned. Default is `None` (false).
/// * `tokenize` - If `Some(true)`, the input texts are tokenized before processing. Default is `None` (true).
///
/// # Returns
///
/// A `PyResult` containing a vector of `AbbreviationDefinition` structs.
#[pyfunction]
#[pyo3(name = "extract_abbreviation_definition_pairs_parallel")]
#[pyo3(signature = (texts, most_common_definition=None, first_definition=None, tokenize=None))]
fn py_extract_abbreviation_definition_pairs_parallel(
    texts: Vec<String>,
    most_common_definition: Option<bool>,
    first_definition: Option<bool>,
    tokenize: Option<bool>,
) -> PyResult<Vec<AbbreviationDefinition>> {
    let options = AbbreviationOptions::new(
        most_common_definition.unwrap_or(false),
        first_definition.unwrap_or(false),
        tokenize.unwrap_or(true),
    );
    Ok(extract_abbreviation_definition_pairs_parallel(
        texts, options,
    ))
}

/// Initializes the Python module.
///
/// This function is called when the Python module is imported. It adds the Python-facing
/// functions to the module.
///
/// # Arguments
///
/// * `m` - The Python module to add the functions to.
///
/// # Returns
///
/// A `PyResult<()>` indicating success or failure of module initialization.
#[pymodule]
fn abbreviation_extractor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        py_extract_abbreviation_definition_pairs,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_extract_abbreviation_definition_pairs_parallel,
        m
    )?)?;
    Ok(())
}
