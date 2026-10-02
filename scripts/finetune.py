"""Fine-tuning LoRA con mlx_lm di un modello installato in oMLX su un dataset scaricato.

Uso:
    ./fine-tune.sh --model Qwen2.5-0.5B-Instruct-4bit --dataset-name banking77 \\
                   --config configs/lora-base.yaml --output adapters/qwen05-banking77

Su un modello quantizzato (es. 4 bit) si tratta di QLoRA. Il config sceglie il prompt di
training e la durata in epoche (vedi configs/lora-base.yaml); le epoche diventano iterazioni
in base alla dimensione del train.

Scrive in <output>/:
    data/{train,valid}.jsonl   esempi in formato chat, generati con il prompt del config
    train_config.yaml          configurazione completa passata a `mlx_lm lora`
    manifest.json              modello, dataset e prompt usati, letto da evaluate.sh
    adapters.safetensors       l'adapter finale (più i checkpoint intermedi)
"""

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rich.console import Console

from common import (ADAPTER_MANIFEST, PROMPT_TYPES, build_messages, load_dataset_dir, resolve_model,
                    warn_if_omlx_running, write_jsonl)

console = Console()

DEFAULTS = {
    "prompt_type": "minimal",
    "batch_size": 8,
    "learning_rate": 1e-4,
    "num_layers": 16,
    "lora": {"rank": 8, "scale": 20.0, "dropout": 0.0},
    "max_seq_length": 1024,
    "seed": 0,
    "steps_per_report": 50,
    "steps_per_eval": 200,
    "save_every": 200,
    "val_batches": -1,
    "mlx_lm": {},
}
DURATION_KEYS = ("epochs", "iters")
POSITIVE_INTS = ("batch_size", "max_seq_length", "steps_per_report", "steps_per_eval", "save_every", "iters")
LORA_KEYS = ("rank", "scale", "dropout")
# Chiavi di mlx_lm che lo script imposta da sé: nella sezione mlx_lm darebbero risultati ambigui.
MANAGED_MLX_KEYS = {"model", "data", "adapter_path", "train", "test", "iters", "batch_size", "learning_rate",
                    "num_layers", "lora_parameters", "max_seq_length", "seed", "steps_per_report",
                    "steps_per_eval", "save_every", "val_batches"}


def load_config(raw: dict | None) -> dict:
    """Valida il config dell'utente e lo completa con i default."""
    raw = raw or {}
    if not isinstance(raw, dict):
        raise SystemExit("Il config deve essere un dizionario YAML.")
    unknown = set(raw) - set(DEFAULTS) - set(DURATION_KEYS)
    if unknown:
        raise SystemExit(f"Chiavi sconosciute nel config: {', '.join(sorted(unknown))}. "
                         "Le opzioni di mlx_lm vanno sotto la chiave mlx_lm.")
    if "epochs" in raw and "iters" in raw:
        raise SystemExit("Nel config indica epochs oppure iters, non entrambi.")

    config = {**DEFAULTS, **raw, "lora": {**DEFAULTS["lora"], **(raw.get("lora") or {})}}
    if "iters" not in config:
        config.setdefault("epochs", 1)

    if config["prompt_type"] not in PROMPT_TYPES:
        raise SystemExit(f"prompt_type deve essere uno fra {', '.join(PROMPT_TYPES)}.")
    for key in POSITIVE_INTS:
        if key in config and (not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0):
            raise SystemExit(f"{key} deve essere un intero positivo.")
    for key in ("epochs", "learning_rate"):
        if key not in config:
            continue
        # PyYAML legge "1e-4" come stringa (YAML 1.1 vuole "1.0e-4"): lo accettiamo comunque.
        try:
            config[key] = float(config[key]) if isinstance(config[key], str) else config[key]
        except ValueError:
            pass
        if not isinstance(config[key], (int, float)) or isinstance(config[key], bool) or config[key] <= 0:
            raise SystemExit(f"{key} deve essere un numero positivo.")
    if set(config["lora"]) - set(LORA_KEYS):
        raise SystemExit(f"lora accetta solo {', '.join(LORA_KEYS)}.")
    if not isinstance(config["mlx_lm"], dict):
        raise SystemExit("mlx_lm deve essere un dizionario.")
    clash = MANAGED_MLX_KEYS & set(config["mlx_lm"])
    if clash:
        raise SystemExit(f"In mlx_lm non usare {', '.join(sorted(clash))}: si impostano al primo livello del config "
                         "o vengono da --model, --dataset-name e --output.")
    return config


def iterations(config: dict, n_train: int) -> int:
    if "iters" in config:
        return config["iters"]
    return max(1, math.ceil(config["epochs"] * n_train / config["batch_size"]))


def mlx_config(config: dict, model_path: Path, data_dir: Path, output: Path, iters: int) -> dict:
    return {
        "model": str(model_path),
        "train": True,
        "data": str(data_dir),
        "fine_tune_type": "lora",
        "mask_prompt": True,  # la loss si calcola solo sulla risposta (l'etichetta)
        "num_layers": config["num_layers"],
        "batch_size": config["batch_size"],
        "iters": iters,
        "learning_rate": config["learning_rate"],
        "lora_parameters": config["lora"],
        "val_batches": config["val_batches"],
        "steps_per_report": config["steps_per_report"],
        "steps_per_eval": config["steps_per_eval"],
        "save_every": config["save_every"],
        "max_seq_length": config["max_seq_length"],
        "adapter_path": str(output),
        "seed": config["seed"],
        **config["mlx_lm"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="nome del modello in oMLX (o cartella del modello)")
    parser.add_argument("--dataset-name", required=True, help="nome locale dato a download.sh")
    parser.add_argument("--config", required=True, type=Path, help="file YAML con i parametri del training")
    parser.add_argument("--output", required=True, type=Path, help="cartella dove salvare l'adapter")
    parser.add_argument("--overwrite", action="store_true", help="riaddestra anche se l'output contiene già un adapter")
    args = parser.parse_args()

    if not args.config.exists():
        raise SystemExit(f"Config {args.config} non trovato.")
    config = load_config(yaml.safe_load(args.config.read_text()))
    dataset = load_dataset_dir(args.dataset_name)
    model_path = resolve_model(args.model)

    output = args.output.resolve()
    if (output / "adapters.safetensors").exists() and not args.overwrite:
        raise SystemExit(f"{args.output} contiene già un adapter: scegli un'altra cartella o aggiungi --overwrite.")

    system = dataset.system_prompt(config["prompt_type"])

    def to_chat(row):
        return {"messages": build_messages(row["text"], system) + [{"role": "assistant", "content": row["label"]}]}

    train, valid = dataset.split("train"), dataset.split("valid")
    data_dir = output / "data"
    write_jsonl(data_dir / "train.jsonl", map(to_chat, train))
    write_jsonl(data_dir / "valid.jsonl", map(to_chat, valid))

    iters = iterations(config, len(train))
    config_file = output / "train_config.yaml"
    config_file.write_text(yaml.safe_dump(mlx_config(config, model_path, data_dir, output, iters), sort_keys=False))

    manifest = {
        "model": args.model,
        "model_path": str(model_path),
        "dataset": dataset.name,
        "dataset_source": dataset.manifest.get("source"),
        "prompt_type": config["prompt_type"],
        "system_prompt": system,
        "config_file": str(args.config.resolve()),
        "config": config,
        "n_train": len(train),
        "n_valid": len(valid),
        "iters": iters,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completed_at": None,
    }
    manifest_file = output / ADAPTER_MANIFEST
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    epochs = f"{config['epochs']} epoche, " if "epochs" in config else ""
    console.print(f"Modello [bold]{args.model}[/], dataset [bold]{dataset.name}[/] ({len(train)} esempi), "
                  f"prompt [bold]{config['prompt_type']}[/]: {epochs}{iters} iterazioni da {config['batch_size']}.")
    console.print(f"Configurazione salvata in {config_file}")
    warn_if_omlx_running(console)

    result = subprocess.run([sys.executable, "-m", "mlx_lm", "lora", "-c", str(config_file)])
    if result.returncode:
        raise SystemExit(f"mlx_lm lora è terminato con codice {result.returncode}.")

    manifest["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
