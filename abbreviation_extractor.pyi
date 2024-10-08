from typing import List, Optional

from sympy import false


class AbbreviationDefinition:
    """
    Represents an abbreviation and its corresponding definition.

    Attributes:
        abbreviation (str): The abbreviated form.
        definition (str): The full form or definition of the abbreviation.
        start (int): The starting index of the definition in the original text.
        end (int): The ending index of the definition in the original text.
    """

    abbreviation: str
    definition: str
    start: int
    end: int

    def __init__(self, abbreviation: str, definition: str, start: int, end: int) -> None:
        """
        Initialize an AbbreviationDefinition instance.

        Args:
            abbreviation (str): The abbreviated form.
            definition (str): The full form or definition of the abbreviation.
            start (int): The starting index of the definition in the original text.
            end (int): The ending index of the definition in the original text.
        """
        ...

    def __repr__(self) -> str:
        """
        Return a string representation of the AbbreviationDefinition instance.

        Returns:
            str: A string representation of the instance.
        """
        ...

class ExtractionError(Exception):
    """
    Represents an error that occurred during the extraction process.

    Attributes:
        error_type (str): The type of error that occurred.
        message (str): A detailed error message.
    """

    def __init__(self, error_type: str, message: str) -> None:
        """
        Initialize an ExtractionError instance.

        Args:
            error_type (str): The type of error that occurred.
            message (str): A detailed error message.
        """
        ...

    def __repr__(self) -> str:
        """
        Return a string representation of the ExtractionError instance.

        Returns:
            str: A string representation of the instance.
        """
        ...

    def __str__(self) -> str:
        """
        Return a string representation of the ExtractionError instance.

        Returns:
            str: A string representation of the instance.
        """
        ...

class ExtractionResult:
    """
    Represents the result of an extraction operation, including both successful extractions and any errors that occurred.

    Attributes:
        extractions (List[AbbreviationDefinition]): A list of successfully extracted abbreviation-definition pairs.
        errors (List[ExtractionError]): A list of errors that occurred during the extraction process.
    """

    extractions: List[AbbreviationDefinition]
    errors: List[ExtractionError]

    def __init__(self, extractions: List[AbbreviationDefinition], errors: List[ExtractionError]) -> None:
        """
        Initialize an ExtractionResult instance.

        Args:
            extractions (List[AbbreviationDefinition]): A list of successfully extracted abbreviation-definition pairs.
            errors (List[ExtractionError]): A list of errors that occurred during the extraction process.
        """
        ...

    def __repr__(self) -> str:
        """
        Return a string representation of the ExtractionResult instance.

        Returns:
            str: A string representation of the instance.
        """
        ...


def extract_abbreviation_definition_pairs(
        text: str,
        most_common_definition: Optional[bool] = None,
        first_definition: Optional[bool] = None,
        tokenize: Optional[bool] = True,
        ignore_errors: Optional[bool] = False
) -> List[AbbreviationDefinition]:
    """
    Extract abbreviation-definition pairs from the given text.

    This function analyzes the input text and identifies abbreviations along with their
    corresponding definitions. It can handle various formats and patterns commonly used
    to introduce abbreviations in text.

    Args:
        text (str): The input text to analyze for abbreviations and definitions.
        most_common_definition (Optional[bool]): If True, return only the most common
            definition for each abbreviation. Defaults to False if not specified.
        first_definition (Optional[bool]): If True, return only the first occurrence of
            each abbreviation-definition pair. Defaults to False if not specified.
        tokenize (Optional[bool]): If True, tokenize the input text before processing.
            Defaults to True if not specified.
        ignore_errors (Optional[bool]): If True, ignore errors during extraction and
            return partial results. Defaults to False if not specified.

    Returns:
        List[AbbreviationDefinition]: A list of AbbreviationDefinition objects, each
        containing an abbreviation, its definition, and the start and end indices of
        the definition in the original text.

    Raises:
        ExtractionError: If an error occurs during extraction and ignore_errors is False.

    Note:
        - If both most_common_definition and first_definition are False or None,
          all found abbreviation-definition pairs will be returned.
        - If both are True, most_common_definition takes precedence.



    Example:
        >>> text = "The World Health Organization (WHO) is a specialized agency."
        >>> result = extract_abbreviation_definition_pairs(text)
        >>> print(result)
        [AbbreviationDefinition(abbreviation='WHO', definition='World Health Organization', start=4, end=29)]
    """
    ...


def extract_abbreviation_definition_pairs_parallel(
        texts: List[str],
        most_common_definition: Optional[bool] = None,
        first_definition: Optional[bool] = None,
        tokenize: Optional[bool] = True
) -> ExtractionResult:
    """
    Extract abbreviation-definition pairs from multiple texts in parallel.

    This function processes multiple input texts concurrently to identify abbreviations
    and their corresponding definitions. It leverages parallel processing to improve
    performance when dealing with large volumes of text.

    Args:
        texts (List[str]): A list of input texts to analyze for abbreviations and definitions.
        most_common_definition (Optional[bool]): If True, return only the most common
            definition for each abbreviation across all texts. Defaults to False if not specified.
        first_definition (Optional[bool]): If True, return only the first occurrence of
            each abbreviation-definition pair across all texts. Defaults to False if not specified.
        tokenize (Optional[bool]): If True, tokenize the input texts before processing.
            Defaults to True if not specified.

    Returns:
         ExtractionResult: An object containing a list of AbbreviationDefinition objects
        (successful extractions) and a list of ExtractionError objects (errors that occurred).

    Note:
        - If both most_common_definition and first_definition are False or None,
          all found abbreviation-definition pairs will be returned.
        - If both are True, most_common_definition takes precedence.
        - The start and end indices in the returned AbbreviationDefinition objects
          are relative to the individual texts, not the combined text.

    Example:
        >>> texts = [
        ...     "The World Health Organization (WHO) is important.",
        ...     "WHO is responsible for international public health."
        ... ]
        >>> result = extract_abbreviation_definition_pairs_parallel(texts)
        >>> print(result)
        [AbbreviationDefinition(abbreviation='WHO', definition='World Health Organization', start=4, end=29)]
    """
    ...

def extract_abbreviations_from_file(
        file_path: str,
        most_common_definition: Optional[bool] = false,
        first_definition: Optional[bool] = false,
        tokenize: Optional[bool] = True,
        chunk_size: Optional[int] = 1024 * 1024,
        num_threads: Optional[int] = None,
        show_progress: Optional[bool] = True
) -> ExtractionResult:
    """
    Extract abbreviation-definition pairs from a file.

    This function reads the input file in chunks, processes them to identify abbreviations
    and their corresponding definitions, and returns the results.

    Args:
        file_path (str): The path to the file to be processed.
        most_common_definition (Optional[bool]): If True, return only the most common
            definition for each abbreviation. Defaults to False if not specified.
        first_definition (Optional[bool]): If True, return only the first occurrence of
            each abbreviation-definition pair. Defaults to False if not specified.
        tokenize (Optional[bool]): If True, tokenize the input text before processing.
            Defaults to True if not specified.
        chunk_size (Optional[int]): The size of chunks to read from the file at a time.
            Defaults to 1 MB (1024 * 1024 bytes).
        num_threads: The number of threads to use for parallel processing. If not specified,
            the number of threads is determined automatically based on the available CPU cores.
        show_progress (Optional[bool]): If True, display a progress bar during processing.
            Defaults to True.

    Returns:
        ExtractionResult: An object containing a list of AbbreviationDefinition objects
        (successful extractions) and a list of ExtractionError objects (errors that occurred).

    Raises:
        ExtractionError: If an error occurs during extraction and ignore_errors is False.

    Note:
        - If both most_common_definition and first_definition are False or None,
          all found abbreviation-definition pairs will be returned.
        - If both are True, most_common_definition takes precedence.
    """
    ...