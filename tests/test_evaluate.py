import json
from pathlib import Path

import pytest

from evaluate import check_adapter, run_name, score, summary_path


def test_score_normalizes_both_sides():
    labels = {"positive", "very negative"}
    assert score("Positive.", "Positive", labels) == (True, True)
    assert score("very negative", "Very Negative", labels) == (True, True)
    assert score("negative", "Positive", labels) == (False, False)
    assert score("very negative", "Positive", labels) == (True, False)


def test_summary_path_and_name():
    assert summary_path(Path("results/base-full.jsonl")) == Path("results/base-full.summary.json")
    assert summary_path(Path("out/run")) == Path("out/run.summary.json")
    assert run_name(Path("results/base-full.jsonl")) == "base-full"
    assert run_name(Path("out/run.txt")) == "run"


def make_adapter(tmp_path, manifest=None):
    (tmp_path / "adapters.safetensors").write_bytes(b"")
    if manifest:
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_check_adapter_requires_weights(tmp_path):
    with pytest.raises(SystemExit, match="adapters.safetensors"):
        check_adapter(tmp_path, Path("/m"), "ds")


def test_check_adapter_rejects_other_model(tmp_path):
    adapter = make_adapter(tmp_path, {"model": "A", "model_path": "/models/A", "dataset": "ds"})
    with pytest.raises(SystemExit, match="addestrato su A"):
        check_adapter(adapter, Path("/models/B"), "ds")
    check_adapter(adapter, Path("/models/A"), "ds")


def test_check_adapter_without_manifest_only_warns(tmp_path):
    check_adapter(make_adapter(tmp_path), Path("/m"), "ds")
