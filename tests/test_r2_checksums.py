import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from write_r2_checksums import write_manifest


def test_checksum_manifest_is_deterministic_and_excludes_invalid_audit_data(tmp_path):
    valid = tmp_path / "result.json"
    valid.write_text("valid\n", encoding="utf-8")
    invalid = tmp_path / "invalid-worker" / "result.json"
    invalid.parent.mkdir()
    invalid.write_text("invalid\n", encoding="utf-8")

    manifest = write_manifest(tmp_path)
    first = manifest.read_text(encoding="utf-8")
    write_manifest(tmp_path)
    assert manifest.read_text(encoding="utf-8") == first
    assert "invalid-worker" not in first
    expected_hash = hashlib.sha256(b"valid\n").hexdigest()
    assert first == f"{expected_hash}  result.json\n"
