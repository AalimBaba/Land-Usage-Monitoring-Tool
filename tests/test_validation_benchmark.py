from __future__ import annotations

from validation_benchmark import run_benchmark


def test_validation_benchmark_reports_expected_builtin_cases():
    report = run_benchmark(include_builtin=True)

    assert report["summary"]["total_cases"] >= 13
    assert report["summary"]["false_accepts"] == 0
    assert report["summary"]["false_rejects"] == 0
    assert report["summary"]["validation_accuracy"] == 1.0
