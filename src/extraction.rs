//! This module contains the core logic for extracting abbreviation-definition pairs from text.
//! It implements the Schwartz-Hearst algorithm for identifying abbreviations and their definitions
//! in biomedical text, as described in:
//!
//! A. Schwartz and M. Hearst (2003) A Simple Algorithm for Identifying Abbreviations Definitions
//! in Biomedical Text. Biocomputing, 451-462.

use crate::utils::{conditions, tokenize_and_clean, PREPOSITIONS};
use crate::Candidate;
use lazy_static::lazy_static;

use crate::abbreviation_definitions::{
    AbbreviationDefinition, AbbreviationOptions, ExtractionError, ExtractionResult,
    FileExtractionOptions,
};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use std::borrow::Cow;
use std::cmp::min;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::panic::AssertUnwindSafe;
use std::sync::{mpsc, Arc, Mutex};
use std::{panic, thread};

lazy_static! {
    /// Regular expression for splitting words on whitespace or hyphens
    static ref WORD_SPLIT_RE: Regex = Regex::new(r"[\s\-]+").unwrap();
    /// Regular expression for cleaning sentences by removing certain punctuation around parentheses
    static ref CLEAN_SENTENCE_RE: Regex = Regex::new(r#"(\()['"\p{Pi}]|['"\p{Pf}]([);:])"#).unwrap();
}

pub fn extract_abbreviation_definition_pairs_wrapper<'a>(
    text: &'a str,
    options: AbbreviationOptions,
) -> Vec<AbbreviationDefinition> {
    if text.is_empty() {
        return Vec::new();
    }
    // Iterate over each sentence in the input text
    // Choose the appropriate iterator based on the 'tokenized' flag
    let sentences: Vec<Cow<'a, str>> = if options.tokenize {
        tokenize_and_clean(text).collect()
    } else {
        text.split('\n').map(Cow::Borrowed).collect()
    };

    // Determine whether to use parallel processing
    let use_parallel = sentences.len() > 50;

    let abbreviations: Vec<AbbreviationDefinition> = if use_parallel {
        // Parallel processing
        sentences
            .par_iter()
            .flat_map(|sentence| process_sentence(sentence))
            .collect()
    } else {
        // Sequential processing
        sentences
            .iter()
            .flat_map(|sentence| process_sentence(sentence))
            .collect()
    };

    // Process the results based on the function parameters
    if options.most_common_definition {
        select_most_common_definitions(abbreviations)
    } else if options.first_definition {
        select_first_definitions(abbreviations)
    } else {
        abbreviations
    }
}

/// Extracts abbreviation-definition pairs from a given text.
///
/// # Arguments
///
/// * `text` - The input text to extract abbreviations from.
/// * `options` - An `AbbreviationOptions` struct containing extraction options:
///   - `most_common_definition`: If true, only the most common definition for each abbreviation is returned.
///   - `first_definition`: If true, only the first definition for each abbreviation is returned.
///   - `tokenized`: If true, the input text is assumed to be pre-tokenized into sentences.
///
/// # Returns
///
/// A vector of `AbbreviationDefinition` structs representing the extracted pairs.
///
/// # Example
///
/// ```
/// use abbreviation_extractor::{extract_abbreviation_definition_pairs, AbbreviationOptions};
///
/// let text = "The World Health Organization (WHO) is a specialized agency. \
///             WHO is responsible for international public health.";
/// let options = AbbreviationOptions::default();
/// let result = extract_abbreviation_definition_pairs(text, options).unwrap();
///
/// assert_eq!(result.len(), 1);
/// assert_eq!(result[0].abbreviation, "WHO");
/// assert_eq!(result[0].definition, "World Health Organization");
/// ```
pub fn extract_abbreviation_definition_pairs(
    text: &str,
    options: AbbreviationOptions,
) -> Result<Vec<AbbreviationDefinition>, ExtractionError> {
    panic::catch_unwind(AssertUnwindSafe(|| {
        extract_abbreviation_definition_pairs_wrapper(text, options)
    }))
    .map_err(|panic_error| {
        let error_msg = panic_error
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| panic_error.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "Unknown panic occurred during abbreviation extraction".to_string());
        ExtractionError::ProcessingError(error_msg)
    })
}

fn process_sentence(sentence: &str) -> Vec<AbbreviationDefinition> {
    let mut abbreviations = Vec::new();
    // Clean the sentence by removing certain punctuation around parentheses
    let sentence = CLEAN_SENTENCE_RE.replace_all(&sentence, "$1$2");
    let sentence = sentence.trim();

    // Find all potential abbreviation candidates in the sentence
    for candidate in best_candidates(&sentence) {
        // get potential synonyms for the candidate (including the candidate itself)
        for potential_synonym in get_potential_synonyms(&candidate) {
            // Try to get a definition for each candidate
            if let Some(definition) = get_definition(&potential_synonym, &sentence) {
                // If a definition is found, try to select the best one
                if let Some(selected_def) = select_definition(&definition, potential_synonym.text())
                {
                    abbreviations.push(AbbreviationDefinition {
                        abbreviation: candidate.text().to_string(),
                        definition: selected_def.text().to_string(),
                        start: selected_def.start(),
                        end: selected_def.stop(),
                    });
                    // if we found a definition, we can break out of the loop and move to the next candidate
                    break;
                }
            }
        }
    }
    abbreviations
}

/// Extracts abbreviation-definition pairs from multiple texts in parallel.
///
/// # Arguments
///
/// * `texts` - A vector of input texts to extract abbreviations from.
/// * `options` - An `AbbreviationOptions` struct containing extraction options:
///   - `most_common_definition`: If true, only the most common definition for each abbreviation is returned.
///   - `first_definition`: If true, only the first definition for each abbreviation is returned.
///   - `tokenized`: If true, the input texts are assumed to be pre-tokenized into sentences.
///
/// # Returns
///
/// A vector of `AbbreviationDefinition` structs representing the extracted pairs from all texts.
///
/// # Example
///
/// ```
/// use abbreviation_extractor::{extract_abbreviation_definition_pairs_parallel, AbbreviationOptions};
///
/// let texts = vec![
///     "The National Aeronautics and Space Administration (NASA) explores space.",
///     "The European Space Agency (ESA) collaborates with NASA.",
///     "Both NASA and ESA conduct important research.",
/// ];
/// let options = AbbreviationOptions::default();
/// let result = extract_abbreviation_definition_pairs_parallel(texts, options);
/// println!("{:?}", result);
/// assert_eq!(result.extractions.len(), 2);
/// assert!(result.extractions.iter().any(|ad| ad.abbreviation == "NASA" && ad.definition == "National Aeronautics and Space Administration"));
/// assert!(result.extractions.iter().any(|ad| ad.abbreviation == "ESA" && ad.definition == "European Space Agency"));
/// ```
pub fn extract_abbreviation_definition_pairs_parallel<T>(
    texts: Vec<T>,
    options: AbbreviationOptions,
) -> ExtractionResult
where
    T: AsRef<str> + Sync,
{
    // Convert texts to Arc for thread-safe sharing
    let texts: Vec<Arc<str>> = texts
        .into_par_iter()
        .map(|t| Arc::from(t.as_ref()))
        .collect();

    // Process each text in parallel
    let results: Vec<Result<Vec<AbbreviationDefinition>, ExtractionError>> = texts
        .par_iter()
        .map(|text| {
            panic::catch_unwind(AssertUnwindSafe(|| {
                // Collect sentences into a Vec<Cow<str>>
                let sentences: Vec<Cow<str>> = if options.tokenize {
                    tokenize_and_clean(text).collect()
                } else {
                    text.split('\n').map(Cow::Borrowed).collect()
                };

                // Process sentences
                sentences
                    .into_par_iter()
                    .flat_map(|sentence| process_sentence(&sentence))
                    .collect()
            }))
            .map_err(|panic_error| {
                let error_msg = panic_error
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| panic_error.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| {
                        "Unknown panic occurred during abbreviation extraction".to_string()
                    });
                ExtractionError::ProcessingError(error_msg)
            })
        })
        .collect();

    // Separate successful extractions and errors
    let mut all_extractions = Vec::new();
    let mut all_errors = Vec::new();

    for result in results {
        match result {
            Ok(extractions) => all_extractions.extend(extractions),
            Err(error) => all_errors.push(error),
        }
    }

    // Apply post-processing based on options
    let final_extractions = if options.most_common_definition {
        select_most_common_definitions(all_extractions)
    } else if options.first_definition {
        select_first_definitions(all_extractions)
    } else {
        all_extractions
    };

    ExtractionResult {
        extractions: final_extractions,
        errors: all_errors,
    }
}

/// Extracts abbreviations from a file safely, using parallel processing.
///
/// This function reads a file in chunks, processes each chunk to extract abbreviations,
/// and handles potential errors and panics that may occur during processing.
///
/// # Arguments
///
/// * `filename` - A string slice that holds the name of the file to process.
/// * `num_threads` - An optional number of threads to use for parallel processing.
///                   If None, it uses the number of available CPU cores.
/// * `options` - An `AbbreviationOptions` struct containing extraction configuration options.
///
/// # Returns
///
/// Returns an `ExtractionResult` which contains:
/// * `extractions`: A vector of successfully extracted `AbbreviationDefinition`s.
/// * `errors`: A vector of `ExtractionError`s that occurred during processing.
///
/// # Errors
///
/// This function will return an `ExtractionResult` with errors if:
/// * The file cannot be opened (IOError).
/// * The thread pool cannot be created (ThreadPoolError).
/// * Any processing errors occur during extraction (ProcessingError).
///
/// # Example
///
/// ```
/// use abbreviation_extractor::{extract_abbreviations_from_file, AbbreviationOptions, FileExtractionOptions};
/// let result = extract_abbreviations_from_file(
///     "input.txt",
///     AbbreviationOptions::default(),
///     FileExtractionOptions::default(),
/// );
/// println!("Extracted {} abbreviations", result.extractions.len());
/// println!("Encountered {} errors", result.errors.len());
/// ```
pub fn extract_abbreviations_from_file(
    filename: &str,
    abbreviation_options: AbbreviationOptions,
    file_extraction_options: FileExtractionOptions,
) -> ExtractionResult {
    // Attempt to open the file
    let file = match File::open(filename) {
        Ok(f) => f,
        Err(e) => {
            return ExtractionResult {
                extractions: Vec::new(),
                errors: vec![ExtractionError::IOError(e.to_string())],
            }
        }
    };

    // Get the file size for the progress bar
    let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);
    let chunk_size = min(file_size, file_extraction_options.chunk_size as u64) as usize;

    // Create a buffered reader with a specified chunk size
    let reader = BufReader::with_capacity(chunk_size, file);

    // Set up a channel for communication between threads
    let (tx, rx) = mpsc::channel();
    let tx = Arc::new(Mutex::new(tx));

    // Determine the number of threads to use
    let thread_count = file_extraction_options.num_threads;

    // Create a thread pool for parallel processing
    let pool = match rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
    {
        Ok(p) => p,
        Err(e) => {
            return ExtractionResult {
                extractions: Vec::new(),
                errors: vec![ExtractionError::ThreadPoolError(e.to_string())],
            }
        }
    };

    // Create a progress bar if show_progress is true
    let pb = if file_extraction_options.show_progress {
        let pb = ProgressBar::new(file_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap_or_else(|e| {
                    eprintln!("Failed to set progress bar template: {}", e);
                    ProgressStyle::default_bar()
                })
                .progress_chars("#>-")
        );
        Some(pb)
    } else {
        None
    };

    // Spawn a thread to read the file and distribute work
    thread::spawn(move || {
        let mut buffer = String::with_capacity(chunk_size);
        let mut bytes_read = 0;

        // Read the file line by line
        for line in reader.lines().filter_map(|line| line.ok()) {
            buffer.push_str(&line);
            buffer.push('\n');
            bytes_read += line.len() as u64 + 1; // Add 1 for the newline character

            // Process the buffer when it reaches or exceeds the chunk size
            if buffer.len() >= chunk_size {
                let chunk_to_process = buffer.clone();
                let chunk_options = abbreviation_options.clone();
                let tx = Arc::clone(&tx);

                // Spawn a task to process the chunk
                pool.spawn(move || {
                    // Use catch_unwind to handle potential panics
                    let result = panic::catch_unwind(AssertUnwindSafe(|| {
                        let mut results = extract_abbreviation_definition_pairs_wrapper(
                            &chunk_to_process,
                            chunk_options,
                        );

                        // Apply filtering options if specified
                        if chunk_options.most_common_definition {
                            results = select_most_common_definitions(results);
                        } else if chunk_options.first_definition {
                            results = select_first_definitions(results);
                        }

                        Ok(results)
                    }));

                    // Handle the result of processing
                    let send_result = match result {
                        Ok(Ok(results)) => Ok(results),
                        Ok(Err(e)) => Err(e),
                        Err(panic_error) => {
                            let error_msg = panic_error
                                .downcast_ref::<&str>()
                                .map(|s| s.to_string())
                                .or_else(|| panic_error.downcast_ref::<String>().cloned())
                                .unwrap_or_else(|| {
                                    "Unknown panic occurred during abbreviation extraction"
                                        .to_string()
                                });
                            Err(ExtractionError::ProcessingError(error_msg))
                        }
                    };

                    // Send the result back through the channel
                    tx.lock().unwrap().send(send_result).unwrap();
                });
                buffer.clear();
            }

            // Update the progress bar if it exists
            if let Some(pb) = &pb {
                pb.set_position(bytes_read);
            }
        }

        // Process any remaining content in the buffer
        if !buffer.is_empty() {
            let result = panic::catch_unwind(AssertUnwindSafe(|| {
                let mut results = extract_abbreviation_definition_pairs_wrapper(
                    &buffer,
                    abbreviation_options.clone(),
                );

                // Apply filtering options to the final chunk
                if abbreviation_options.most_common_definition {
                    results = select_most_common_definitions(results);
                } else if abbreviation_options.first_definition {
                    results = select_first_definitions(results);
                }

                Ok(results)
            }));

            // Handle the result of processing the final chunk
            let send_result = match result {
                Ok(Ok(results)) => Ok(results),
                Ok(Err(e)) => Err(e),
                Err(panic_error) => {
                    let error_msg = panic_error
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| panic_error.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| {
                            "Unknown panic occurred during abbreviation extraction".to_string()
                        });
                    Err(ExtractionError::ProcessingError(error_msg))
                }
            };

            // Send the final result through the channel
            tx.lock().unwrap().send(send_result).unwrap();
        }

        // Signal that we're done sending chunks by dropping the sender
        drop(tx);

        // Finish the progress bar if it exists
        if let Some(pb) = pb {
            pb.finish_with_message("File processing completed");
        }
    });

    // Collect results from all processed chunks
    let mut extractions = Vec::new();
    let mut errors = Vec::new();

    // Receive results from the channel
    for result in rx {
        match result {
            Ok(chunk_results) => extractions.extend(chunk_results),
            Err(e) => errors.push(e),
        }
    }

    // Return the final ExtractionResult
    ExtractionResult {
        extractions,
        errors,
    }
}

/// Selects the most common definitions for each abbreviation.
///
/// # Arguments
///
/// * `abbrevs` - A vector of `AbbreviationDefinition` structs.
///
/// # Returns
///
/// A vector of `AbbreviationDefinition` structs with only the most common definition for each abbreviation.
fn select_most_common_definitions(
    abbrevs: Vec<AbbreviationDefinition>,
) -> Vec<AbbreviationDefinition> {
    // Create a nested HashMap to count occurrences of each definition for each abbreviation
    let mut definition_counts: FxHashMap<String, FxHashMap<String, usize>> = FxHashMap::default();

    // Count the occurrences of each definition for each abbreviation
    for abbrev in &abbrevs {
        definition_counts
            .entry(abbrev.abbreviation.clone())
            .or_insert_with(FxHashMap::default)
            .entry(abbrev.definition.clone())
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }

    // Create a HashMap to store the most common definition for each abbreviation
    let mut most_common: FxHashMap<String, AbbreviationDefinition> = FxHashMap::default();

    // Find the most common definition for each abbreviation
    for abbrev in abbrevs {
        if let Some(counts) = definition_counts.get(&abbrev.abbreviation) {
            if let Some(max_count) = counts.values().max() {
                if counts.get(&abbrev.definition) == Some(max_count) {
                    most_common
                        .entry(abbrev.abbreviation.clone())
                        .or_insert(abbrev);
                }
            }
        }
    }

    // Convert the HashMap values to a Vec
    most_common.into_values().collect()
}

/// Selects the first definition for each abbreviation.
///
/// # Arguments
///
/// * `abbrevs` - A vector of `AbbreviationDefinition` structs.
///
/// # Returns
///
/// A vector of `AbbreviationDefinition` structs with only the first definition for each abbreviation
fn select_first_definitions(abbrevs: Vec<AbbreviationDefinition>) -> Vec<AbbreviationDefinition> {
    // Create an FxHashSet to keep track of abbreviations we've seen
    let mut seen = FxHashSet::default();

    // Filter the input vector, keeping only the first occurrence of each abbreviation
    abbrevs
        .into_iter()
        .filter(|abbrev| seen.insert(abbrev.abbreviation.clone()))
        .collect()
}

/// Generates potential synonyms for a given abbreviation candidate.
///
/// This function creates a list of potential synonyms for an abbreviation candidate,
/// which can be used to improve the chances of finding a correct definition.
///
/// # Arguments
///
/// * `candidate` - A reference to the original `Candidate` struct representing the abbreviation.
///
/// # Returns
///
/// A vector of `Candidate` structs representing potential synonyms, including the original candidate
/// if its length is 10 or less.
///
/// ```
fn get_potential_synonyms<'a>(candidate: &'a Candidate<'a>) -> Vec<Candidate<'a>> {
    let mut candidates = Vec::with_capacity(3);
    let text = candidate.text();

    // Only add the original candidate if its length is 10 or less
    if text.len() <= 10 {
        candidates.push(candidate.clone());
    }

    let words: Vec<&str> = text.split_whitespace().collect();

    if words.len() == 2 {
        let (first_word, second_word) = (words[0], words[1]);
        // if any is less than 3, return
        if first_word.len() < 3 || second_word.len() < 3 {
            return candidates;
        }

        let first_word_upper_count = first_word.chars().filter(|c| c.is_uppercase()).count();
        let second_word_upper_count = second_word.chars().filter(|c| c.is_uppercase()).count();

        let first_word_ratio = first_word_upper_count as f32 / first_word.len() as f32;
        let second_word_ratio = second_word_upper_count as f32 / second_word.len() as f32;

        if first_word_ratio >= 0.5 && second_word_ratio < 0.1 {
            candidates.push(Candidate::new(
                Cow::Owned(first_word.to_string()),
                candidate.start(),
                candidate.stop(),
            ));
        } else if second_word_ratio >= 0.5 && first_word_ratio < 0.1 {
            candidates.push(Candidate::new(
                Cow::Owned(second_word.to_string()),
                candidate.start(),
                candidate.stop(),
            ));
        }
    }

    candidates
}

/// Finds the best abbreviation candidates in a given sentence.
///
/// # Arguments
///
/// * `sentence` - The input sentence to search for abbreviation candidates.
///
/// # Returns
///
/// A vector of `Candidate` structs representing potential abbreviations.
pub fn best_candidates(sentence: &str) -> Vec<Candidate> {
    // Convert the sentence to bytes for efficient character comparisons
    let sent_bytes = sentence.as_bytes();

    // Check if '(' is present in the sentence. If not, return an empty vector
    if !sent_bytes.contains(&b'(') {
        return Vec::new();
    }

    let mut close_index = 0;
    let mut candidates: Vec<Candidate> = Vec::new();

    // Main loop to find candidates
    loop {
        if close_index >= sent_bytes.len() {
            break;
        }

        // Look for open parenthesis. Need leading whitespace to avoid matching mathematical and chemical formulae
        let open_index = sent_bytes[close_index..]
            .windows(2)
            .position(|window| window == b" (")
            .map(|pos| pos + close_index);

        match open_index {
            Some(open_index) => {
                // Advance beyond whitespace in ' ('
                let open_index = open_index + 1;

                // Look for close parenthesis
                close_index = open_index + 1;
                let mut open_count = 1;
                let mut skip = false;

                // Find matching closing parenthesis
                while open_count != 0 {
                    let char = match sent_bytes.iter().nth(close_index) {
                        Some(c) => c,
                        None => {
                            // We found an opening bracket but no associated closing bracket
                            // Skip the opening bracket
                            skip = true;
                            break;
                        }
                    };
                    if *char == b'(' {
                        open_count += 1;
                    } else if [b')', b';', b':'].contains(&char) {
                        open_count -= 1;
                    }
                    close_index += 1;
                }

                if skip {
                    close_index = open_index + 1;
                    continue;
                }

                if close_index <= 0 {
                    break;
                }

                // Extract the candidate from within the parentheses
                let start = open_index + 1;
                let stop = close_index - 1;
                let candidate_text = safe_slice(sentence, start, stop);

                // Take into account whitespace that should be removed
                let start = start + candidate_text.len() - candidate_text.trim_start().len();
                let stop = stop - candidate_text.len() + candidate_text.trim_end().len();
                let candidate = safe_slice(sentence, start, stop);

                // Check if the candidate meets certain conditions
                if conditions(&candidate) {
                    candidates.push(Candidate::new(candidate.to_string(), start, stop));
                }
            }
            None => break, // No more opening parentheses found, exit the loop
        }
    }
    candidates // Return the list of valid candidates
}

/// Attempts to find a definition for a given abbreviation candidate in a sentence.
///
/// # Arguments
///
/// * `candidate` - The abbreviation candidate.
/// * `sentence` - The sentence containing the candidate.
///
/// # Returns
///
/// An `Option<Candidate>` representing the potential definition, if found.
pub fn get_definition<'a>(candidate: &Candidate<'a>, sentence: &'a str) -> Option<Candidate<'a>> {
    // Convert the part of the sentence before the candidate to lowercase
    let lowercase_sentence = sentence[..candidate.start().saturating_sub(2)].to_lowercase();

    // Split the lowercase sentence into tokens (words)
    let tokens: Vec<&str> = WORD_SPLIT_RE.split(&lowercase_sentence).collect();

    // Get the first character of the candidate (the key we're looking for)
    let key = candidate
        .text()
        .chars()
        .next()
        .and_then(|c| c.to_lowercase().next())
        .unwrap_or('\0'); // Default to null character if no key is found

    // Create a vector of the first characters of each token
    let first_chars: Vec<char> = tokens.iter().filter_map(|t| t.chars().next()).collect();

    // Count how many times the key appears at the start of tokens in the definition
    let definition_freq = first_chars.iter().filter(|&&c| c == key).count();

    // Count how many times the key appears in the candidate
    let candidate_freq = candidate
        .text()
        .to_lowercase()
        .chars()
        .filter(|&c| c == key)
        .count();

    // If there are fewer occurrences of the key in the definition than in the candidate, return None
    if candidate_freq > definition_freq {
        return None;
    }

    let mut count: isize = 0;
    let mut start: i32 = 0;
    let mut start_index: isize = (first_chars.len() - 1) as isize;

    // Add a maximum iteration count to prevent infinite loops
    let max_iterations = first_chars.len() * 2;
    let mut iterations = 0;

    // Look for a sequence of tokens that could form the definition
    while count < candidate_freq as isize && iterations < max_iterations {
        // If we've searched beyond the beginning of the tokens, return None
        if start.abs() > first_chars.len() as i32 || start_index < 0 {
            return None;
        }
        start -= 1;

        // Look for the key in the definition, starting from the end
        let slice_start = first_chars.len().saturating_add_signed(start as isize);
        if let Some(position) = first_chars[slice_start..].iter().position(|&c| c == key) {
            start_index = (slice_start + position) as isize;

            // Check if the potential definition starts with a preposition or conjunction
            let sniffer_start = tokens[..start_index as usize].join(" ").len();
            // extract up to 4 characters from the sentence starting at sniffer_start, handling Unicode also
            let preposition = match sentence.get(sniffer_start..) {
                Some(s) => s
                    .char_indices()
                    .take(4)
                    .map(|(_, c)| c)
                    .collect::<String>()
                    .trim()
                    .to_string(),
                None => String::new(),
            };
            if PREPOSITIONS.is_match(&preposition) {
                // If it does, adjust our search
                start -= 1;
                if start_index == 0 {
                    return None;
                }
                start_index -= 1;
            }
        }

        // Count the number of keys in the current potential definition
        if start_index < 0 {
            break;
        }
        count = first_chars[start_index as usize..]
            .iter()
            .filter(|&&c| c == key)
            .count() as isize;
        iterations += 1;
    }

    if start_index < 0 || iterations >= max_iterations || start_index as usize >= tokens.len() {
        return None;
    }

    // We found enough keys in the definition, so construct the definition candidate
    let start = tokens[..start_index as usize].join(" ").len();
    let stop = candidate.start() - 1;
    let mut candidate_text = safe_slice(sentence, start, stop);

    // Remove leading and trailing whitespace
    let mut start = start + candidate_text.len() - candidate_text.trim_start().len();
    let stop = stop - candidate_text.len() + candidate_text.trim_end().len();

    if !best_candidates(safe_slice(sentence, start, stop)).is_empty() {
        return None;
    }

    // if char before start is an hyphen, then take all character before the hyphen till we hit the start of space
    if (sentence.chars().nth(start) == Some('-') || sentence.chars().nth(start) == Some(')'))
        && start > 0
    {
        let mut hyphen_index = start - 1;
        while hyphen_index > 0 && sentence.chars().nth(hyphen_index - 1) != Some(' ') {
            hyphen_index -= 1;
        }
        start = hyphen_index;
    }
    candidate_text = safe_slice(sentence, start, stop);

    // Return the definition as a new Candidate
    Some(Candidate::new(candidate_text, start, stop))
}

/// Selects the best definition for a given abbreviation.
///
/// # Arguments
///
/// * `definition` - The potential definition candidate.
/// * `abbrev` - The abbreviation string.
///
/// # Returns
///
/// An `Option<Candidate>` representing the selected definition, if valid.
pub fn select_definition<'a>(definition: &'a Candidate<'a>, abbrev: &str) -> Option<Candidate<'a>> {
    // Check if abbreviation is longer than definition or if it's a full word in the definition
    if definition.text().len() < abbrev.len()
        || definition
            .text()
            .split_whitespace()
            .any(|word| word == abbrev)
    {
        return None;
    }

    // Convert abbreviation to lowercase and collect characters
    let abbrev_lowercase = abbrev.to_ascii_lowercase();
    let abbrev_chars: Vec<char> = abbrev_lowercase.chars().collect();

    // Collect characters of the definition
    let def_chars: Vec<char> = definition.text().chars().collect();

    // Initialize indices to start from the end of both strings
    let mut s_index: isize = (abbrev_chars.len() - 1) as isize;
    let mut l_index: isize = (def_chars.len() - 1) as isize;

    let max_iterations = definition.text().len() + abbrev.len();
    let mut iterations = 0;

    // Main loop for matching characters
    loop {
        // Exit loop if we've reached the start of either string
        if l_index < 0 || s_index < 0 || iterations >= max_iterations {
            break;
        }

        // Get current characters from both strings
        let long_char = def_chars[l_index as usize].to_ascii_lowercase();
        let short_char = abbrev_chars[s_index as usize].to_ascii_lowercase();

        if !short_char.is_alphanumeric() {
            // If abbreviation character is not alphanumeric, move to next
            s_index -= 1;
        } else if s_index == 0 {
            // We've reached the start of the abbreviation,
            // and the last character before short char is not a '('. for example, "human (h) Upf1 protein (p)", "hUpf1p", we don't stop at "h) Upf1 protein (p)"
            if short_char == long_char && (l_index == 0 || def_chars[l_index as usize - 1] != '(') {
                // If characters match and we're at a word boundary or start of definition, we're done
                if l_index == 0 || !def_chars[l_index as usize - 1].is_alphanumeric() {
                    break;
                } else {
                    // Otherwise, keep looking
                    l_index -= 1;
                }
            } else {
                // If characters don't match, keep looking in definition
                l_index -= 1;
            }
        } else {
            // We're not at the start of the abbreviation
            if short_char == long_char {
                // If characters match, move both indices
                s_index -= 1;
                l_index -= 1;
            } else {
                // If they don't match, only move definition index
                l_index -= 1;
            }
        }
        iterations += 1;
    }

    // If we've exhausted either string without finding a match, return None
    if l_index < 0 || s_index < 0 || iterations >= max_iterations {
        return None;
    }

    l_index = walk_backwards(&def_chars, l_index);

    // Create a new candidate with the matched portion of the definition
    let new_candidate = Candidate::new(
        utf8_slice_start(definition.text(), l_index as usize),
        definition.start(),
        definition.stop(),
    );

    // Ensure the definition doesn't start with a preposition
    let candidate = if !PREPOSITIONS.is_match(new_candidate.text()) {
        new_candidate
    } else {
        definition.clone()
    };

    // Count the number of tokens in the candidate
    let tokens = candidate.text().split_whitespace().count();
    let length = abbrev.len();

    // Check if the definition is too long compared to the abbreviation
    if tokens > min(length + 5, length * 2) {
        return None;
    }

    // Check for unbalanced parentheses in the candidate
    if candidate.text().chars().filter(|&c| c == '(').count()
        != candidate.text().chars().filter(|&c| c == ')').count()
    {
        return None;
    }

    // ensure the definition does not start with a preposition
    if PREPOSITIONS.is_match(candidate.text()) {
        return None;
    }

    // If all checks pass, return the candidate
    Some(candidate)
}

/// Walks backwards through a vector of characters to find the start of a definition.
///
/// This function is used to refine the starting point of a definition by handling
/// special cases such as hyphens, slashes, and chemical formulas.
///
/// # Arguments
///
/// * `def_chars` - A reference to a vector of characters representing the definition.
/// * `start` - The initial starting index from which to walk backwards.
///
/// # Returns
///
/// An `isize` representing the new starting index after walking backwards.
///
/// # Details
///
/// The function performs the following steps:
/// 1. If the start index is 0, it returns 0 immediately.
/// 2. If the character before the start index is a hyphen or slash, it walks back
///    to the beginning of the word.
/// 3. If it encounters a potential chemical formula, it walks back
///    to include the entire formula.
///
fn walk_backwards(def_chars: &Vec<char>, start: isize) -> isize {
    if start == 0 {
        return 0;
    }
    let mut index = start;

    // if l_index > 0 and l_index - 1 is an hyphen or a '/', then take an character before the hyphen till we hit the start of space
    if def_chars[index as usize - 1] == '-' || def_chars[index as usize - 1] == '/' {
        while index > 0 {
            if def_chars[index as usize - 1] == ' ' {
                break;
            }
            index -= 1;
        }
    }

    // if it's a chemical formula, then we can walk back
    if index as usize >= 2
        && def_chars[index as usize].is_numeric()
        && def_chars[index as usize - 2] == ','
    {
        index -= 1;
        while index > 0 {
            if def_chars[index as usize - 1] == ' ' {
                break;
            }
            index -= 1;
        }
    }

    index
}

fn safe_slice(s: &str, start: usize, end: usize) -> &str {
    s.get(start..end).unwrap_or("")
}

fn utf8_slice_start(s: &str, start_char_index: usize) -> &str {
    match s.char_indices().nth(start_char_index) {
        Some((byte_index, _)) => &s[byte_index..],
        None => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_best_candidates() {
        let sentence = "The World Health Organization (WHO) is a specialized agency.";
        let candidates = best_candidates(sentence);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0], Candidate::new("WHO".to_string(), 31, 34));

        let sentence = "The National Aeronautics and Space Administration (NASA) explores space.";
        let candidates = best_candidates(sentence);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0], Candidate::new("NASA".to_string(), 51, 55));
    }

    #[test]
    fn test_multiple_candidates() {
        let sentence = "The United Nations (UN) and World Health Organization (WHO) work together.";
        let candidates = best_candidates(sentence);

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].text(), "UN");
        assert_eq!(candidates[1].text(), "WHO");
    }

    #[test]
    fn test_no_candidates() {
        let sentence = "This sentence has no abbreviations.";
        let candidates = best_candidates(sentence);

        assert_eq!(candidates.len(), 0);
    }

    #[test]
    fn test_invalid_candidates() {
        let sentence = "Invalid candidates: (A) (toolong) (123)";
        let candidates = best_candidates(sentence);

        assert_eq!(candidates.len(), 0);
    }

    #[test]
    fn test_get_definition_simple() {
        let sentence = "The World Health Organization (WHO) is a specialized agency.";
        let candidate = Candidate::new("WHO".to_string(), 31, 34);
        let definition = get_definition(&candidate, sentence);

        assert!(definition.is_some());
        let def = definition.unwrap();
        // println!("{:?}", &sentence[31..34]);
        assert_eq!(
            def,
            Candidate::new("World Health Organization".to_string(), 4, 29)
        );

        let sentence = "The National Aeronautics and Space Administration (NASA) explores space.";
        let candidate = Candidate::new("NASA".to_string(), 51, 55);
        let definition = get_definition(&candidate, sentence);
        // println!("{:?}", definition);
        assert!(definition.is_some());
        let def = definition.unwrap();
        assert_eq!(
            def,
            Candidate::new(
                "National Aeronautics and Space Administration".to_string(),
                4,
                49
            )
        );
    }

    #[test]
    fn test_get_definition_with_preposition() {
        let sentence = "We use the Rust Programming Language (RPL) for systems programming.";
        let candidate = Candidate::new("RPL".to_string(), 38, 41);
        let definition = get_definition(&candidate, sentence);

        assert!(definition.is_some());
        let def = definition.unwrap();
        assert_eq!(
            def,
            Candidate::new("Rust Programming Language".to_string(), 11, 36)
        );
    }

    #[test]
    fn test_get_definition_no_match() {
        let sentence = "This sentence contains (XYZ) but no matching definition.";
        let candidate = Candidate::new("XYZ".to_string(), 24, 27);
        let definition = get_definition(&candidate, sentence);

        assert!(definition.is_none());
    }

    #[test]
    fn test_get_definition_case_insensitive() {
        let sentence = "The central processing unit (CPU) is the brain of a computer.";
        let candidate = Candidate::new("CPU".to_string(), 29, 32);
        let definition = get_definition(&candidate, sentence);

        assert!(definition.is_some());
        let def = definition.unwrap();
        assert_eq!(def.text(), "central processing unit");
    }

    #[test]
    fn test_select_definition_simple() {
        let definition = Candidate::new("World Health Organization".to_string(), 4, 29);
        let abbrev = "WHO";
        let result = select_definition(&definition, abbrev);
        assert!(result.is_some());
        let selected = result.unwrap();
        assert_eq!(
            selected,
            Candidate::new("World Health Organization".to_string(), 4, 29)
        );

        let definition = Candidate::new(
            "National Aeronautics and Space Administration".to_string(),
            4,
            49,
        );
        let abbrev = "NASA";
        let result = select_definition(&definition, abbrev);
        assert!(result.is_some());
        let selected = result.unwrap();
        assert_eq!(
            selected,
            Candidate::new(
                "National Aeronautics and Space Administration".to_string(),
                4,
                49
            )
        );
    }

    #[test]
    fn test_select_definition_partial() {
        let definition = Candidate::new("World Health Organization".to_string(), 4, 29);
        // let definition = Candidate::new("World Health Organization is a specialized agency".to_string(), 4, 29);
        // let definition = Candidate::new("The World Health Organization is a specialized agency".to_string(), 0, 54);
        let abbrev = "WHO";
        let result = select_definition(&definition, abbrev);
        assert!(result.is_some());
        let selected = result.unwrap();
        assert_eq!(
            selected,
            Candidate::new("World Health Organization".to_string(), 4, 29)
        );
    }

    #[test]
    fn test_select_definition_no_match() {
        let definition = Candidate::new("United Nations".to_string(), 0, 14);
        let abbrev = "WHO";
        let result = select_definition(&definition, abbrev);
        assert!(result.is_none());
    }

    #[test]
    fn test_select_definition_abbreviation_in_definition() {
        let definition = Candidate::new("World WHO Organization".to_string(), 0, 22);
        let abbrev = "WHO";
        let result = select_definition(&definition, abbrev);
        assert!(result.is_none());
    }

    #[test]
    fn test_select_definition_full_word() {
        let definition = Candidate::new("The WHO is a specialized agency".to_string(), 0, 34);
        let abbrev = "WHO";
        let result = select_definition(&definition, abbrev);
        assert!(
            result.is_none(),
            "Should return None when abbreviation is a full word in the definition"
        );
    }

    fn assert_abbreviation(
        result: &[AbbreviationDefinition],
        abbreviation: &str,
        definition: &str,
    ) {
        assert!(
            result
                .iter()
                .any(|ad| ad.abbreviation == abbreviation && ad.definition == definition),
            "Failed to find abbreviation '{}' with definition '{}'",
            abbreviation,
            definition
        );
    }

    #[test]
    fn test_extract_abbreviation_definition_pairs() {
        let text = "The World Health Organization (WHO) is a specialized agency. \
                    WHO is responsible for international public health.";
        let result =
            extract_abbreviation_definition_pairs(text, AbbreviationOptions::default()).unwrap();

        assert_eq!(result.len(), 1);
        assert_abbreviation(&result, "WHO", "World Health Organization");

        let text = "The National Aeronautics and Space Administration (NASA) explores space.";
        let result =
            extract_abbreviation_definition_pairs(text, AbbreviationOptions::default()).unwrap();
        assert_eq!(result.len(), 1);
        assert_abbreviation(
            &result,
            "NASA",
            "National Aeronautics and Space Administration",
        );

        let text = "Wiskott-Aldrich syndrome protein (WASP)";
        let result =
            extract_abbreviation_definition_pairs(text, AbbreviationOptions::default()).unwrap();
        assert_eq!(result.len(), 1);
        assert_abbreviation(&result, "WASP", "Wiskott-Aldrich syndrome protein");
    }

    #[test]
    fn test_extract_multiple_abbreviations() {
        let text =
            "The United Nations (UN) works closely with the World Health Organization (WHO). \
                    Both UN and WHO are international organizations.";
        let result =
            extract_abbreviation_definition_pairs(text, AbbreviationOptions::default()).unwrap();

        assert_eq!(result.len(), 2);
        assert_abbreviation(&result, "UN", "United Nations");
        assert_abbreviation(&result, "WHO", "World Health Organization");
    }

    #[test]
    fn test_most_common_definition() {
        let text = "The World Health Organization (WHO) is important. \n\
                    The World Heritage Organization (WHO) is different. \n\
                    The World Health Organization (WHO) is a UN agency.";
        let options = AbbreviationOptions::new(true, false, false);
        let result = extract_abbreviation_definition_pairs(text, options).unwrap();
        assert_eq!(result.len(), 1);
        assert_abbreviation(&result, "WHO", "World Health Organization");
    }

    #[test]
    fn test_first_definition() {
        let text = "The World Heritage Organization (WHO) is important. \
                    The World Health Organization (WHO) is different.";
        let options = AbbreviationOptions::new(false, true, false);
        let result = extract_abbreviation_definition_pairs(text, options).unwrap();

        assert_eq!(result.len(), 1);
        assert_abbreviation(&result, "WHO", "World Heritage Organization");
    }

    fn run_extraction_test(
        text: &str,
        expected_pairs: Vec<(&str, &str)>,
        options: AbbreviationOptions,
    ) {
        let result = extract_abbreviation_definition_pairs(text, options).unwrap();
        // println!("result is {:?}", result);
        for (acronym, expected_term) in expected_pairs {
            assert_abbreviation(&result, acronym, expected_term);
        }
    }

    #[test]
    fn test_extract_abbreviations() {
        let text = r#"The endoplasmic reticulum (ER) in Saccharomyces cerevisiae consists of a
        reticulum underlying the plasma membrane (cortical ER) and ER associated with
        the nuclear envelope (nuclear ER).
        The SH3 domain of Myo5p regulates the
        polymerization of actin through interactions with both Las17p, a homolog of
        mammalian  Wiskott-Aldrich syndrome protein (WASP), and Vrp1p, a homolog of
        WASP-interacting protein (WIP).
        Ribonuclease P (RNase P) is a ubiquitous endoribonuclease that cleaves precursor
        tRNAs to generate mature 5prime prime or minute termini.
        The purified proteins
        were separated by sodium dodecyl sulfate-polyacrylamide gel electrophoresis (SDS-PAGE) and
        identified by peptide mass fingerprint analysis using
        matrix-assisted laser desorption/ionization (MALDI) mass spectrometry."#;

        let options = AbbreviationOptions::default();
        run_extraction_test(
            text,
            vec![
                ("ER", "endoplasmic reticulum"),
                ("WASP", "Wiskott-Aldrich syndrome protein"),
                ("WIP", "WASP-interacting protein"),
                ("RNase P", "Ribonuclease P"),
                (
                    "SDS-PAGE",
                    "sodium dodecyl sulfate-polyacrylamide gel electrophoresis",
                ),
                ("MALDI", "matrix-assisted laser desorption/ionization"),
            ],
            options,
        );
    }

    #[test]
    fn test_extract_abbreviations_with_special_cases() {
        let text = r#"Theory of mind (ToM; Smith 2009) broadly refers to humans' ability to represent the mental states of others,
        including their desires, beliefs, and intentions.
        Applications of text-to-speech (TTS) include:
        We review astronomy and physics engagement with the
        Open Researcher and Contributor iD (ORCID) service as a solution.
        The Proceeds of Crime Act 2002 ("PoCA 2002")."#;

        let options = AbbreviationOptions::default();

        run_extraction_test(
            text,
            vec![
                ("ToM", "Theory of mind"),
                ("TTS", "text-to-speech"),
                ("ORCID", "Open Researcher and Contributor iD"),
                ("PoCA 2002", "Proceeds of Crime Act 2002"),
            ],
            options,
        );
    }

    #[test]
    fn test_extract_abbreviations_with_edge_cases() {
        let text = r#"The "satellite" goal of the program was accomplished when China established a space presence with the launch of Dongfanghong I in 1970; although, it wasn't until the 21st century that the PRC space program kicked into high gear, with the rapid development, buildup and deployment of rockets, satellites, and the first Taikonaut (astronaut) in October 2003. In fact, prior to 2010, the PRC had only conducted ten space launches, one of which put the satellite into orbit.
        Once more, also for the Space Race, a strong transatlantic link could strengthen the path towards a peaceful and prosperous future for humankind and by consequence, a more secure period for our democracies: it is in our hands (and brains) to transform these ideas into a great reality.
        Berlin is acknowledging the vulnerabilities that could potentially arise through hostile acts in space and set up its own space monitoring center, called the Air and Space Operations Center (ASOC) in September 2020 ."#;

        let options = AbbreviationOptions::default();
        run_extraction_test(
            text,
            vec![("ASOC", "Air and Space Operations Center")],
            options,
        );

        let result = extract_abbreviation_definition_pairs(text, options).unwrap();
        assert!(!result.iter().any(|ad| ad.abbreviation == "astronaut"));
        assert!(!result.iter().any(|ad| ad.abbreviation == "and brains"));
    }

    #[test]
    fn test_extract_abbreviations_with_edge_cases_2() {
        let text = r#"this approach, which we term high-throughput mass spectrometric protein complex
identification (HMS-PCI). Beginning with 10% of predicted yeast proteins as.
The Rep78 and Rep68 proteins of adeno-associated virus (AAV) type 2 are involved
in DNA replication, regulation of gene expression, and targeting site-specific
integration.
Ligand-receptor interaction for other C19-steroids was also examined. While
5-androstene-3beta, 17beta-diol (ADIOL) displayed estrogenic activity in this
system,
The Ogg1 protein of Saccharomyces cerevisiae belongs to a family of DNA
glycosylases and apurinic/apyrimidinic site (AP) lyases, the signature of which
is the alpha-helix. We have used the yeast three-hybrid system (D. J. SenGupta, B. Zhang, B.
Kraemer, P. Pochart, S. Fields, and M. Wickens, Proc. Natl. Acad. Sci. USA
93:8496-8501, 1996) to study binding of the human immunodeficiency virus type 1
(HIV-1) Gag protein to the HIV-1 RNA encapsidation signal (HIVPsi). Interaction
of these elements results in the activation of a reporter gene in the yeast
Saccharomyces cerevisiae. Using this system, we have shown that the HIV-1 Gag
Department of Chemistry and Biochemistry, Texas Tech University, Lubbock, TX,
79409-1061, USA. u0nes@ttacs.ttu.edu

Sterol C-methylations catalyzed by the (S)-adenosyl-L-methionine:
Delta(24)-sterol methyl transferase (SMT) have provided the focus for study of
electrophilic alkylations, a reaction type of functional importance in C-C bond
formation of natural products."#;

        let options = AbbreviationOptions::new(false, false, true);
        run_extraction_test(
            text,
            vec![
                (
                    "HMS-PCI",
                    "high-throughput mass spectrometric protein complex identification",
                ),
                ("AAV", "adeno-associated virus"),
                ("ADIOL", "5-androstene-3beta, 17beta-diol"),
                ("AP", "apurinic/apyrimidinic site"),
                ("HIV-1", "human immunodeficiency virus type 1"),
                ("HIVPsi", "HIV-1 RNA encapsidation signal"),
                ("SMT", "Delta(24)-sterol methyl transferase"),
            ],
            options,
        );

        let text = r#"cells, NMD appears to involve splicing-dependent alterations to mRNA as well as
ribosome-associated components of the translational apparatus. To date, human
(h) Upf1 protein (p) (hUpf1p), a group 1 RNA helicase named after its
Saccharomyces cerevisiae orthologue that functions in both translation
termination and NMD, has been the only factor shown to be required for NMD in
mammalian cells. Here, we describe human orthologues to
binding sites for Ro60 and La proteins, and Ro RNPs are thus physiologically
proteins and recombinant hY (rhY) co-expressed in yeast, we found that RNPs
made of rRo60/rhY/rLa were readily reassembled. Reconstitution of tripartite
RNPs was critically dependent on the presence of an appropriate Ro60 binding
encodes a membrane protein. The bait is expressed in its natural environment,
the membrane, whereas the protein partner (the prey) is fused to a cytoplasmic
he transactivational properties of tamoxifen in a basic yeast model system
which reconstitutes ligand-dependent human estrogen receptor-alpha (hER alpha)
gene activation. Tamoxifen exerted low agonist activity in this system compared
calculated by fitting experimental data with a logistic dose-response function.
domain and phosphatidylserines. For this purpose, mixed bilayers of 1-palmitoyl,
2-oleoyl-sn-glycero-3-phosphocholine (POPC) and
"#;
        let options = AbbreviationOptions::new(false, false, true);
        run_extraction_test(
            text,
            vec![
                ("hUpf1p", "human (h) Upf1 protein (p)"),
                ("rhY", "recombinant hY"),
                ("hER alpha", "human estrogen receptor-alpha"),
                ("POPC", "1-palmitoyl, 2-oleoyl-sn-glycero-3-phosphocholine"),
            ],
            options,
        );
    }

    #[test]
    fn test_parallel_extraction_multiple_texts_str() {
        let texts = vec![
            "The National Aeronautics and Space Administration (NASA) explores space.",
            "The European Space Agency (ESA) collaborates with NASA.",
            "Both NASA and ESA conduct important research.",
        ];
        let options = AbbreviationOptions::default();
        let result = extract_abbreviation_definition_pairs_parallel(texts, options);
        assert_eq!(result.extractions.len(), 2);
        assert_abbreviation(
            &result.extractions,
            "NASA",
            "National Aeronautics and Space Administration",
        );
        assert_abbreviation(&result.extractions, "ESA", "European Space Agency");
    }

    #[test]
    fn test_parallel_extraction_multiple_texts_string() {
        let texts = vec![
            "The National Aeronautics and Space Administration (NASA) explores space.".to_string(),
            "The European Space Agency (ESA) collaborates with NASA.".to_string(),
            "Both NASA and ESA conduct important research.".to_string(),
        ];
        let options = AbbreviationOptions::default();
        let result = extract_abbreviation_definition_pairs_parallel(texts, options);
        assert_eq!(result.extractions.len(), 2);
        assert_abbreviation(
            &result.extractions,
            "NASA",
            "National Aeronautics and Space Administration",
        );
        assert_abbreviation(&result.extractions, "ESA", "European Space Agency");
    }

    #[test]
    fn test_tokenize_and_clean() {
        let input = r#"First sentence.
Second sentence with a
newline in the middle.

Third sentence after an empty line.
Fourth sentence.
Fifth sentence with trailing newline.
"#;

        let expected = vec![
            "First sentence.",
            "Second sentence with a newline in the middle.",
            "Third sentence after an empty line.",
            "Fourth sentence.",
            "Fifth sentence with trailing newline.",
        ];

        let result: Vec<String> = tokenize_and_clean(input)
            .map(|cow| cow.into_owned())
            .collect();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_unicode_chars() {
        let text = r#"Two kinds of mechanical valve, St. Jude Medical (SJM) and Björk-Shiley (B-S), in patients with single valve replacement have been evaluated on a view point of intravascular hemolysis.
The World Health Organization (WHO) works globally. La Société Nationale des Chemins de fer Français (SNCF) est l'entreprise ferroviaire publique française.,
Em português, a Organização Mundial da Saúde (OMS) é muito importante.,
Всемирная организация здравоохранения (Воз) работает во всем мире.",
The Société Générale des Surveillances (SGS) is a multinational company.,
Το Ινστιτούτο Τεχνολογίας Υπολογιστών και Εκδόσεων (ΙΤΥΕ) είναι ερευνητικός οργανισμός.",
"#;
        let options = AbbreviationOptions::default();

        run_extraction_test(
            text,
            vec![
                ("SJM", "St. Jude Medical"),
                ("B-S", "Björk-Shiley"),
                ("WHO", "World Health Organization"),
                ("SNCF", "Société Nationale des Chemins de fer Français"),
                ("OMS", "Organização Mundial da Saúde"),
                ("Воз", "Всемирная организация здравоохранения"),
                ("SGS", "Société Générale des Surveillances"),
                ("ΙΤΥΕ", "Ινστιτούτο Τεχνολογίας Υπολογιστών και Εκδόσεων"),
            ],
            options,
        );
    }

    #[test]
    fn test_extract_abbreviations_from_file_safe() {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("benches");
        path.push("pubmed_abstracts_20240801_to_20240809.txt");

        let abbreviation_options = AbbreviationOptions::new(true, false, true);
        let file_extraction_options = FileExtractionOptions {
            num_threads: num_cpus::get(),
            chunk_size: 1024 * 1024,
            show_progress: false,
        };
        let result = extract_abbreviations_from_file(
            path.to_str().unwrap(),
            abbreviation_options,
            file_extraction_options,
        );

        // Check that we have some successful results
        assert!(
            !result.extractions.is_empty(),
            "No abbreviations were extracted"
        );

        // assert > 6200
        assert!(
            result.extractions.len() > 6200,
            "Expected more than 6200 abbreviations, found {}",
            result.extractions.len()
        );
    }
}
