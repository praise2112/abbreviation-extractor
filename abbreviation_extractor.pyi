from typing import List, Optional


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


def extract_abbreviation_definition_pairs(
        text: str,
        most_common_definition: Optional[bool] = None,
        first_definition: Optional[bool] = None,
        tokenize: Optional[bool] = True
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

    Returns:
        List[AbbreviationDefinition]: A list of AbbreviationDefinition objects, each
        containing an abbreviation, its definition, and the start and end indices of
        the definition in the original text.

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
) -> List[AbbreviationDefinition]:
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
        List[AbbreviationDefinition]: A list of AbbreviationDefinition objects, each
        containing an abbreviation, its definition, and the start and end indices of
        the definition in the original text.

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
