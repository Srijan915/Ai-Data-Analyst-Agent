from app.core.offline_benchmark import benchmark_cases, run_offline_benchmark


def test_offline_benchmark_is_reproducible_and_complete():
    assert len(benchmark_cases()) == 50
    assert run_offline_benchmark() == {"total": 50, "passed": 50, "failed": 0}
