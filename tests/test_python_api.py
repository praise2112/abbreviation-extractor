import pickle

import pytest
import abbreviation_extractor as ae

TEXT = "The World Health Organization (WHO) is a specialized agency."


def test_results_survive_pickling():
    result = ae.extract_abbreviation_definition_pairs_parallel([TEXT])
    restored = pickle.loads(pickle.dumps(result))
    assert repr(restored) == repr(result)
    [pair] = pickle.loads(pickle.dumps(result.extractions))
    assert (pair.abbreviation, pair.definition) == ("WHO", "World Health Organization")


def test_missing_file_error_is_readable_and_picklable():
    [error] = ae.extract_abbreviations_from_file("/nonexistent/file", show_progress=False).errors
    assert isinstance(error, ae.ExtractionError.IOError)
    assert str(error).startswith("IO error: ")
    assert repr(pickle.loads(pickle.dumps(error))) == repr(error)


def test_corrupt_pickle_raises_value_error():
    pair = ae.AbbreviationDefinition("A", "b", 0, 1)
    with pytest.raises(ValueError):
        pair.__setstate__(b"\x01")
