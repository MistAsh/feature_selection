import pytest

from tests.feature_selection.core.test_data import TOTAL_TEST_CASES


@pytest.mark.parametrize(
    'algo_case',
    TOTAL_TEST_CASES,
    ids=[case['test_name'] for case in TOTAL_TEST_CASES]
)
def test_feature_selector(algo_case):

    assert algo_case['selector_class'] is not None

    selector = algo_case['selector_class'](**algo_case['selector_params'])

    try:
        selector.fit(**algo_case['fit_data'])
    except Exception as e:
        pytest.fail(f"fit raised an exception: {e}")

    result = selector.transform(**algo_case['fit_data'])

    assert isinstance(result, tuple), \
        "transform must return tuple (X, y)"

    assert len(selector.history) > 0
