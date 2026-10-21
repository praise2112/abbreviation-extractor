use punkt_n::params::{
    DefinesNonPrefixCharacters, DefinesNonWordCharacters, DefinesPunctuation,
    DefinesSentenceEndings,
};
use punkt_n::{SentenceTokenizer, TrainingData};
use regex::Regex;
use std::borrow::Cow;
use std::sync::LazyLock;

/// This regex matches strings that have the first word as a preposition
pub static PREPOSITIONS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^\s*(and|a[st]|[io]n|of|for|to|with|by|at|from|in|on|over|under|or|like)\b").unwrap());
/// This regex matches strings that are all lowercase and 7 or more characters long
pub static LONG_LOWER_CASE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\p{Ll}{7,}$").unwrap());
/// Sentence tokenizer for English text
static TOKENIZATION_DATA: LazyLock<TrainingData> = LazyLock::new(TrainingData::english);
static WHITESPACE_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());

/// The character sets of punkt's `Standard` parameters, checked without hashing.
struct StandardCharacters;

impl DefinesSentenceEndings for StandardCharacters {
    fn is_sentence_ending(c: &char) -> bool {
        matches!(c, '.' | '?' | '!')
    }
}

impl DefinesNonWordCharacters for StandardCharacters {
    fn is_nonword_char(c: &char) -> bool {
        matches!(
            c,
            '?' | '!' | ')' | '"' | ';' | '}' | ']' | '*' | ':' | '@' | '\'' | '(' | '{' | '['
        )
    }
}

impl DefinesPunctuation for StandardCharacters {
    fn is_punctuation(c: &char) -> bool {
        matches!(c, ';' | ':' | ',' | '.' | '!' | '?')
    }
}

impl DefinesNonPrefixCharacters for StandardCharacters {
    fn is_nonprefix_char(c: &char) -> bool {
        matches!(
            c,
            '(' | '"' | '`' | '{' | '[' | ':' | ';' | '&' | '#' | '*' | '@' | ')' | '}' | ']' | '-' | ','
        )
    }
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
    SentenceTokenizer::<StandardCharacters>::new(text, &TOKENIZATION_DATA).map(|sent| {
        if sent.contains('\n') {
            Cow::Owned(WHITESPACE_REGEX.replace_all(sent.trim(), " ").into_owned())
        } else {
            Cow::Borrowed(sent.trim())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::StandardCharacters;
    use punkt_n::params::{
        DefinesNonPrefixCharacters, DefinesNonWordCharacters, DefinesPunctuation,
        DefinesSentenceEndings, Standard,
    };

    #[test]
    fn standard_characters_match_punkt_standard_sets() {
        for c in (0..0x3000).filter_map(char::from_u32) {
            assert_eq!(StandardCharacters::is_sentence_ending(&c), Standard::is_sentence_ending(&c), "{c:?}");
            assert_eq!(StandardCharacters::is_nonword_char(&c), Standard::is_nonword_char(&c), "{c:?}");
            assert_eq!(StandardCharacters::is_punctuation(&c), Standard::is_punctuation(&c), "{c:?}");
            assert_eq!(StandardCharacters::is_nonprefix_char(&c), Standard::is_nonprefix_char(&c), "{c:?}");
        }
    }
}
