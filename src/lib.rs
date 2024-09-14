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
//! ## Basic Usage
//!
//! ### Rust
//!
//! ```rust
//! use abbreviation_extractor::{extract_abbreviation_definition_pairs, AbbreviationOptions};
//!
//! let text = "The World Health Organization (WHO) is a specialized agency.";
//! let options = AbbreviationOptions::default();
//! let result = extract_abbreviation_definition_pairs(text, options).unwrap();
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
//! ## Parallel Processing
//!
//! For processing multiple texts in parallel, you can use the `extract_abbreviation_definition_pairs_parallel` function:
//!
//! ### Rust
//!
//! ```rust
//! use abbreviation_extractor::{extract_abbreviation_definition_pairs_parallel, AbbreviationOptions};
//!
//! let texts = vec![
//!     "The World Health Organization (WHO) is a specialized agency.",
//!     "The United Nations (UN) works closely with WHO.",
//!     "The European Union (EU) is a political and economic union.",
//! ];
//!
//! let options = AbbreviationOptions::default();
//! let result = extract_abbreviation_definition_pairs_parallel(texts, options);
//!
//! for extraction in result.extractions {
//!     println!("Abbreviation: {}, Definition: {}", extraction.abbreviation, extraction.definition);
//! }
//! ```
//!
//! ### Python
//!
//! ```python
//! from abbreviation_extractor import extract_abbreviation_definition_pairs_parallel
//!
//! texts = [
//!     "The World Health Organization (WHO) is a specialized agency.",
//!     "The United Nations (UN) works closely with WHO.",
//!     "The European Union (EU) is a political and economic union.",
//! ]
//!
//! result = extract_abbreviation_definition_pairs_parallel(texts)
//!
//! for extraction in result.extractions:
//! print(f"Abbreviation: {extraction.abbreviation}, Definition: {extraction.definition}")
//! ```
//!
//! ## Processing Large Files
//!
//! For extracting abbreviations from large files, you can use the `extract_abbreviations_from_file` function:
//!
//! ### Rust
//!
//! ```rust
//! use abbreviation_extractor::{extract_abbreviations_from_file, AbbreviationOptions, FileExtractionOptions};
//!
//! let file_path = "path/to/your/large/file.txt";
//! let abbreviation_options = AbbreviationOptions::default();
//! let file_options = FileExtractionOptions::default();
//!
//! let result = extract_abbreviations_from_file(file_path, abbreviation_options, file_options);
//!
//! for extraction in result.extractions {
//!     println!("Abbreviation: {}, Definition: {}", extraction.abbreviation, extraction.definition);
//! }
//! ```
//!
//! ### Python
//!
//! ```python
//! from abbreviation_extractor import extract_abbreviations_from_file
//!
//! file_path = "path/to/your/large/file.txt"
//! result = extract_abbreviations_from_file(file_path)
//!
//! for extraction in result.extractions:
//! print(f"Abbreviation: {extraction.abbreviation}, Definition: {extraction.definition}")
//! ```
//!
//! You can customize the file extraction process by specifying additional parameters:
//!
//! ```python
//! result = extract_abbreviations_from_file(
//!     file_path,
//!     most_common_definition=True,
//!     first_definition=False,
//!     tokenize=True,
//!     num_threads=4,
//!     show_progress=True,
//!     chunk_size=2048 * 1024  # 2MB chunks
//! )
//! ```
//!
//! ## Functions
//!
//! The main functions provided by this library are:
//!
//! - [`extract_abbreviation_definition_pairs`]: Extracts abbreviation-definition pairs from a single text.
//! - [`extract_abbreviation_definition_pairs_parallel`]: Extracts abbreviation-definition pairs from multiple texts in parallel.
//! - [`extract_abbreviations_from_file`]: Extracts abbreviation-definition pairs from a large file.
//!
//! For detailed information on each function, please refer to their individual documentation.
//!
//! ## Structs/Enums
//! - [`AbbreviationOptions`]: Defines the `AbbreviationOptions` struct for customizing abbreviation extraction
//! - [`FileExtractionOptions`]: Defines the `FileExtractionOptions` struct for customizing file extraction for [`extract_abbreviations_from_file`]
//! - [`AbbreviationDefinition`]: Defines the `AbbreviationDefinition` struct for storing abbreviation-definition pairs
//! - [`ExtractionResult`]: Defines the `ExtractionResult` struct returned by [`extract_abbreviation_definition_pairs_parallel`] and [`extract_abbreviations_from_file`]
//! - [`ExtractionError`]: Defines the `ExtractionError` enum for error handling
//!
//! ## Modules
//!
//! - [`candidate`]: Defines the `Candidate` struct used in the extraction process
//! - [`extraction`]: Contains the core logic for extracting abbreviation-definition pairs
//! - [`utils`]: Utility functions and regular expressions used in the extraction process
//! - [`abbreviation_definitions`]: Defines the `AbbreviationDefinition` and `AbbreviationOptions` structs

use log::{error, warn};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub mod abbreviation_definitions;
pub mod candidate;
pub mod extraction;
pub mod utils;

pub use abbreviation_definitions::{
    AbbreviationDefinition, AbbreviationOptions, ExtractionError, ExtractionResult,
    FileExtractionOptions,
};
pub use candidate::Candidate;
pub use extraction::{
    best_candidates, extract_abbreviation_definition_pairs,
    extract_abbreviation_definition_pairs_parallel, extract_abbreviations_from_file,
    get_definition, select_definition,
};

fn handle_extraction_error(
    e: ExtractionError,
    ignore_errors: bool,
) -> PyResult<Vec<AbbreviationDefinition>> {
    let error_msg = match e {
        ExtractionError::ProcessingError(msg) => format!("Extraction error: {}", msg),
        ExtractionError::IOError(msg) => format!("IO error: {}", msg),
        ExtractionError::ThreadPoolError(msg) => format!("Thread pool error: {}", msg),
    };
    if ignore_errors {
        warn!("Ignoring error: {}", error_msg);
        Ok(Vec::new())
    } else {
        error!("{}", error_msg);
        Err(PyRuntimeError::new_err(error_msg))
    }
}

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
/// * `ignore_errors` - If `Some(false)`, errors during extraction are ignored and an empty vector is returned.
///
/// # Returns
///
/// A `PyResult` containing a vector of `AbbreviationDefinition` structs.
#[pyfunction]
#[pyo3(name = "extract_abbreviation_definition_pairs")]
#[pyo3(signature = (text, most_common_definition=None, first_definition=None, tokenize=None, ignore_errors=None)
)]
fn py_extract_abbreviation_definition_pairs(
    text: &str,
    most_common_definition: Option<bool>,
    first_definition: Option<bool>,
    tokenize: Option<bool>,
    ignore_errors: Option<bool>,
) -> PyResult<Vec<AbbreviationDefinition>> {
    let options = AbbreviationOptions::new(
        most_common_definition.unwrap_or(false),
        first_definition.unwrap_or(false),
        tokenize.unwrap_or(true),
    );
    let ignore_errors = ignore_errors.unwrap_or(false);
    match extract_abbreviation_definition_pairs(text, options) {
        Ok(v) => Ok(v),
        Err(e) => handle_extraction_error(e, ignore_errors),
    }
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
/// A `PyResult` containing an `ExtractionResult` struct.
#[pyfunction]
#[pyo3(name = "extract_abbreviation_definition_pairs_parallel")]
#[pyo3(signature = (texts, most_common_definition=None, first_definition=None, tokenize=None))]
fn py_extract_abbreviation_definition_pairs_parallel(
    texts: Vec<String>,
    most_common_definition: Option<bool>,
    first_definition: Option<bool>,
    tokenize: Option<bool>,
) -> PyResult<ExtractionResult> {
    let options = AbbreviationOptions::new(
        most_common_definition.unwrap_or(false),
        first_definition.unwrap_or(false),
        tokenize.unwrap_or(true),
    );
    Ok(extract_abbreviation_definition_pairs_parallel(
        texts, options,
    ))
}

/// Extracts abbreviation-definition pairs from a file.
///
/// This function is exposed to Python and serves as a wrapper around the Rust
/// `extract_abbreviations_from_file` function.
///
/// # Arguments
///
/// * `file_path` - The path to the file to extract abbreviations from.
/// * `most_common_definition` - If `Some(true)`, only the most common definition for each
///   abbreviation is returned. Default is `None` (false).
/// * `first_definition` - If `Some(true)`, only the first definition for each abbreviation
///   is returned. Default is `None` (false).
/// * `tokenize` - If `Some(true)`, the input text is tokenized before processing. Default is `None` (true).
/// * `chunk_size` - The size of chunks to read from the file at a time. Default is `1024 * 1024` (1MB).
/// * `num_threads` - The number of threads to use for parallel processing. Default is `num of cpus`.
/// * `show_progress` - If `Some(true)`, a progress bar is displayed during extraction. Default is `None` (true).
///
/// # Returns
///
/// A `PyResult` containing an `ExtractionResult` struct.
#[pyfunction]
#[pyo3(name = "extract_abbreviations_from_file")]
#[pyo3(signature = (file_path, most_common_definition=None, first_definition=None, tokenize=None, num_threads=None, show_progress=None, chunk_size=None))]
fn py_extract_abbreviations_from_file(
    file_path: String,
    most_common_definition: Option<bool>,
    first_definition: Option<bool>,
    tokenize: Option<bool>,
    num_threads: Option<usize>,
    show_progress: Option<bool>,
    chunk_size: Option<usize>,
) -> PyResult<ExtractionResult> {
    let abbreviation_options = AbbreviationOptions::new(
        most_common_definition.unwrap_or(false),
        first_definition.unwrap_or(false),
        tokenize.unwrap_or(true),
    );

    let file_options = FileExtractionOptions::new(
        num_threads.unwrap_or(num_cpus::get()),
        chunk_size.unwrap_or(1024 * 1024), // Default to 1MB if not specified
        show_progress.unwrap_or(true),
    );
    Ok(extract_abbreviations_from_file(
        &file_path,
        abbreviation_options,
        file_options,
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
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(
        py_extract_abbreviation_definition_pairs,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_extract_abbreviation_definition_pairs_parallel,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_extract_abbreviations_from_file, m)?)?;
    Ok(())
}
