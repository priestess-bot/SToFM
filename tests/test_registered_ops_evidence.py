import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from verify_stofm_registered_ops_evidence import verify_suite


def test_committed_fp32_registered_operator_evidence_is_complete():
    result = verify_suite(ROOT / "benchmark-results" / "r3-v100-registered-ops-fp32-20260816")
    assert result["status"] == "passed"
    assert result["precision"] == "fp32"
    assert result["trial_count"] == 3


def test_committed_fp16_registered_operator_evidence_is_complete():
    result = verify_suite(ROOT / "benchmark-results" / "r3-v100-registered-ops-fp16-20260816")
    assert result["status"] == "passed"
    assert result["precision"] == "fp16"
    assert result["trial_count"] == 3
