"""Structure-only tests for the offline SQL security evaluation cases."""

from evals.run_security_eval import SAFE_CASES, UNSAFE_CASES


def test_security_eval_case_lists_are_labeled_and_non_empty():
    assert UNSAFE_CASES
    assert SAFE_CASES
    assert all(case.get("label") for case in UNSAFE_CASES + SAFE_CASES)
