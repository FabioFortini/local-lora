"""Fine-tuning LoRA (QLoRA, il modello è a 4 bit) con mlx_lm.

Uso:
    uv run scripts/finetune.py                       # parametri di default
    uv run scripts/finetune.py --iters 300           # prova veloce
    uv run scripts/finetune.py --rank 16 --lr 5e-5 --adapter-path adapters/r16

Scrive la configurazione usata in <adapter-path>/train_config.yaml e lancia
`python -m mlx_lm lora`. L'adapter finale finisce in <adapter-path>/adapters.safetensors.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from common import ADAPTERS_DIR, DEFAULT_MODEL, LORA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default=str(ADAPTERS_DIR / "banking77"))
    # ~9.000 esempi di training: 1.200 iterazioni x batch 8 ≈ 1 epoca.
    parser.add_argument("--iters", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-layers", type=int, default=16, help="layer da adattare (-1 = tutti)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not (LORA_DIR / "train.jsonl").exists():
        raise SystemExit("Dati di training non trovati: lancia prima `uv run scripts/download.py`.")

    config = {
        "model": args.model,
        "train": True,
        "data": str(LORA_DIR),
        "fine_tune_type": "lora",
        "mask_prompt": True,  # la loss si calcola solo sulla risposta (l'etichetta)
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "iters": args.iters,
        "learning_rate": args.lr,
        "lora_parameters": {"rank": args.rank, "scale": args.scale, "dropout": args.dropout},
        "val_batches": -1,
        "steps_per_report": 50,
        "steps_per_eval": 200,
        "save_every": 200,
        "max_seq_length": 1024,
        "adapter_path": args.adapter_path,
        "seed": args.seed,
    }

    out = Path(args.adapter_path)
    out.mkdir(parents=True, exist_ok=True)
    config_file = out / "train_config.yaml"
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Configurazione salvata in {config_file}", flush=True)

    subprocess.run([sys.executable, "-m", "mlx_lm", "lora", "-c", str(config_file)], check=True)


if __name__ == "__main__":
    main()
