use lazy_static::lazy_static;
use punkt_n::params::Standard;
use punkt_n::{SentenceTokenizer, TrainingData};
use regex::Regex;
use std::borrow::Cow;

lazy_static! {
    /// This regex matches strings that have the first word as a preposition
    pub static ref PREPOSITIONS: Regex = Regex::new(r"(?i)^\s*(and|a[st]|[io]n|of|for|to|with|by|at|from|in|on|over|under|or|like)\b").unwrap();
    /// This regex matches strings that are all lowercase and 7 or more characters long
    pub static ref LONG_LOWER_CASE: Regex = Regex::new(r"^\p{Ll}{7,}$").unwrap();
    /// Sentence tokenizer for English text
    static ref TOKENIZATION_DATA: TrainingData = TrainingData::english();

}

/// Checks if a given candidate string meets the criteria for being a valid abbreviation.
///
/// The conditions are:
/// 1. Length between 2 and 10 characters
/// 2. No more than 2 words
/// 3. Not all lowercase if longer than 6 characters
/// 4. Contains at least one letter
/// 5. Starts with an alphanumeric character
pub fn conditions(candidate: &str) -> bool {
    let length = candidate.len();

    // Quick check: Ensure the length is between 2 and 10 characters
    // This is the most common disqualifier, so we check it first for efficiency
    if length < 2 || length > 12 {
        // if length < 2 || length > 10 {
        return false;
    }

    // Check that the candidate has no more than 2 words
    // This is a relatively quick check that can eliminate many candidates early
    let word_count = candidate.split_whitespace().count();
    if word_count > 2 {
        return false;
    }

    // Check if the candidate is all lowercase and longer than 6 characters
    // We use a pre-compiled regex for efficiency in repeated calls
    if LONG_LOWER_CASE.is_match(candidate) {
        return false;
    }

    // Perform a single pass through the string for remaining checks
    // This is more efficient than multiple separate loops
    let mut has_letter = false;
    for (i, c) in candidate.chars().enumerate() {
        if i == 0 && !c.is_alphanumeric() {
            // The first character must be alphanumeric
            return false;
        }
        if c.is_alphabetic() {
            // We've found a letter, so we can stop checking
            has_letter = true;
            break;
        }
    }

    // cannot start the abbrv with a preposition
    if word_count != 1 && PREPOSITIONS.is_match(candidate) {
        return false;
    }

    // If we've made it this far and found a letter, the candidate is valid
    has_letter
}

/// Tokenizes and cleans a given text into sentences.
///
/// This function takes a string of text, tokenizes it into sentences, and cleans each sentence
/// by trimming whitespace and replacing newlines with spaces.
///
/// # Arguments
///
/// * `text` - A string slice containing the text to be tokenized and cleaned.
///
/// # Returns
///
/// An iterator that yields `Cow<str>` items, where each item represents a cleaned sentence.
///
/// # Details
///
/// The function performs the following steps:
/// 1. Uses `SentenceTokenizer` to split the text into sentences.
/// 2. For each sentence:
///    - Trims leading and trailing whitespace.
///    - Replaces newlines with spaces if present.
///    - Returns the sentence as a `Cow<str>` (either borrowed or owned, depending on whether modifications were needed).
///
/// # Examples
///
/// ```
/// use crate::abbreviation_extractor::utils::tokenize_and_clean;
///
/// let text = "First sentence.\nSecond sentence\nwith a newline.";
/// let sentences: Vec<String> = tokenize_and_clean(text)
///     .map(|cow| cow.into_owned())
///     .collect();
///
/// assert_eq!(sentences, vec![
///     "First sentence.",
///     "Second sentence with a newline."
/// ]);
/// ```
pub fn tokenize_and_clean<'a>(text: &'a str) -> impl Iterator<Item = Cow<'a, str>> + 'a {
    SentenceTokenizer::<Standard>::new(text, &TOKENIZATION_DATA).map(|sent| {
        if sent.contains('\n') {
            Cow::Owned(sent.replace('\n', " ").trim().to_string())
        } else {
            Cow::Borrowed(sent.trim())
        }
    })
}
