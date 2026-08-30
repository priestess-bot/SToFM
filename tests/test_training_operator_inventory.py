from benchmarks.train_stofm_fake_flagos import _markdown_report


def test_training_report_mentions_known_gap_solutions(tmp_path):
    result = {
        "status": "passed",
        "strict": True,
        "steps": [
            {
                "step": 0,
                "loss": 2.0,
                "mcm_loss": 1.0,
                "pdr_loss": 1.0,
                "max_grad": 1.0,
                "step_ms": 3.0,
            }
        ],
        "initial_parameter_sha256": "a",
        "final_parameter_sha256": "b",
        "checkpoint": str(tmp_path / "checkpoint.pt"),
        "operator_inventory": {"fallback_compute_ops": []},
    }
    report = _markdown_report(result)
    assert "cosine_embedding_loss" in report
    assert "foreach" in report
    assert "AMP" in report
    assert "fallback：无" in report
