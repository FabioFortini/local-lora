from pathlib import Path

import pytest
import yaml

from finetune import iterations, load_config, mlx_config


def test_defaults_and_one_epoch():
    config = load_config(None)
    assert config["prompt_type"] == "minimal" and config["epochs"] == 1 and "iters" not in config
    assert config["lora"] == {"rank": 8, "scale": 20.0, "dropout": 0.0}


def test_epochs_become_iterations():
    assert iterations(load_config({"epochs": 1, "batch_size": 8}), 8994) == 1125
    assert iterations(load_config({"epochs": 0.5, "batch_size": 4}), 10) == 2
    assert iterations(load_config({"iters": 30}), 8994) == 30


def test_epochs_and_iters_are_exclusive():
    with pytest.raises(SystemExit, match="non entrambi"):
        load_config({"epochs": 1, "iters": 10})


def test_unknown_keys_are_rejected():
    with pytest.raises(SystemExit, match="lerning_rate"):
        load_config({"lerning_rate": 1e-4})


def test_learning_rate_written_as_yaml_1_1_string():
    config = load_config(yaml.safe_load("learning_rate: 1e-4"))
    assert config["learning_rate"] == pytest.approx(1e-4)


@pytest.mark.parametrize("raw", [{"prompt_type": "long"}, {"batch_size": 0}, {"epochs": -1},
                                 {"learning_rate": "fast"}, {"lora": {"alpha": 1}}, {"mlx_lm": []}])
def test_invalid_values(raw):
    with pytest.raises(SystemExit):
        load_config(raw)


def test_mlx_passthrough_cannot_override_managed_keys():
    with pytest.raises(SystemExit, match="iters"):
        load_config({"mlx_lm": {"iters": 5}})
    config = load_config({"lora": {"rank": 16}, "mlx_lm": {"grad_checkpoint": True}})
    out = mlx_config(config, Path("/m"), Path("/d"), Path("/o"), 100)
    assert out["grad_checkpoint"] is True and out["iters"] == 100 and out["mask_prompt"] is True
    assert out["lora_parameters"] == {"rank": 16, "scale": 20.0, "dropout": 0.0}


def test_example_configs_are_valid():
    root = Path(__file__).resolve().parent.parent / "configs"
    for f in root.glob("*.yaml"):
        load_config(yaml.safe_load(f.read_text()))
