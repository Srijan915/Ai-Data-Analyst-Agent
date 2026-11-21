"""Structure-only tests for the offline grounding evaluation cases."""

from evals.run_grounding_eval import build_grounding_cases


def test_grounding_eval_builds_thirty_labeled_cases():
    cases = build_grounding_cases()

    assert len(cases) == 30
    assert all(case.label for case in cases)
    assert all(case.correct_explanation for case in cases)
    assert all(case.corrupted_explanation for case in cases)
