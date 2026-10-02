import json

import pytest

import common
from common import Dataset, find_models, normalize, resolve_model


def make_model(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    return path


def test_normalize_strips_punctuation_and_case():
    assert normalize('  "Card_Arrival." ') == "card_arrival"
    assert normalize("`positive`") == "positive"
    assert normalize("card arrival") == "card arrival"  # nessuna correzione oltre la pulizia


def test_find_models_nested_like_omlx(tmp_path):
    make_model(tmp_path / "mlx-community" / "Qwen-0.5B")
    make_model(tmp_path / "flat-model")
    (tmp_path / "mlx-community" / "not-a-model").mkdir()
    assert set(find_models([tmp_path])) == {"mlx-community/Qwen-0.5B", "flat-model"}


def test_resolve_model_by_leaf_name_full_name_and_path(tmp_path):
    model = make_model(tmp_path / "mlx-community" / "Qwen-0.5B")
    assert resolve_model("Qwen-0.5B", [tmp_path]) == model
    assert resolve_model("mlx-community/Qwen-0.5B", [tmp_path]) == model
    assert resolve_model(str(model), []) == model.resolve()


def test_resolve_model_ambiguous_leaf_name(tmp_path):
    make_model(tmp_path / "org-a" / "Qwen")
    make_model(tmp_path / "org-b" / "Qwen")
    with pytest.raises(SystemExit, match="ambiguo"):
        resolve_model("Qwen", [tmp_path])
    assert resolve_model("org-b/Qwen", [tmp_path]) == tmp_path / "org-b" / "Qwen"


def test_resolve_model_missing_lists_installed(tmp_path):
    make_model(tmp_path / "org" / "Installed")
    with pytest.raises(SystemExit, match="org/Installed"):
        resolve_model("Missing", [tmp_path])


def test_system_prompt_substitutes_labels():
    ds = Dataset("x", None, {}, ["a", "b"], {"minimal": "short", "full": "pick:\n{labels}\nend"})
    assert ds.system_prompt("minimal") == "short"
    assert ds.system_prompt("full") == "pick:\na\nb\nend"
    with pytest.raises(ValueError):
        ds.system_prompt("other")


@pytest.fixture
def datasets_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATASETS_DIR", tmp_path)
    return tmp_path


def write_dataset(root, name, task="classification", prompts="minimal: m\nfull: 'f {labels}'\n"):
    d = root / name
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"task": task}))
    (d / "labels.json").write_text(json.dumps(["a", "b"]))
    (d / "prompts.yaml").write_text(prompts)
    return d


def test_load_dataset_dir(datasets_dir):
    write_dataset(datasets_dir, "toy")
    ds = common.load_dataset_dir("toy")
    assert ds.labels == ["a", "b"]
    assert ds.system_prompt("full") == "f a\nb"


def test_load_dataset_dir_missing_lists_available(datasets_dir):
    write_dataset(datasets_dir, "toy")
    with pytest.raises(SystemExit, match="Disponibili: toy"):
        common.load_dataset_dir("other")


def test_load_dataset_dir_rejects_unsupported_task(datasets_dir):
    write_dataset(datasets_dir, "qa", task="qa")
    with pytest.raises(SystemExit, match="non supportato"):
        common.load_dataset_dir("qa")


def test_load_dataset_dir_requires_both_prompts(datasets_dir):
    write_dataset(datasets_dir, "toy", prompts="minimal: m\n")
    with pytest.raises(SystemExit, match="full"):
        common.load_dataset_dir("toy")


def test_dataset_name_validation():
    with pytest.raises(SystemExit):
        common.dataset_dir("../escape")
