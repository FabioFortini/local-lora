"""Scarica mteb/banking77 e prepara gli split per valutazione e fine-tuning.

Uso:
    uv run scripts/download.py [--valid-ratio 0.1] [--train-prompt minimal|full] [--seed 42]

Produce:
    data/labels.json             le 77 etichette, ordinate per id
    data/raw/{train,valid,test}.jsonl   {"text", "label"}
    data/lora/{train,valid}.jsonl       {"messages": [...]} per mlx_lm lora

Il validation set è ricavato dal train: il test resta intoccato fino alla valutazione.
"""

import argparse
import json
import random

from datasets import load_dataset
from rich.console import Console
from rich.table import Table

from common import LABELS_FILE, LORA_DIR, RAW_DIR, build_messages, write_jsonl

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument(
        "--train-prompt",
        choices=["minimal", "full"],
        default="minimal",
        help="system prompt usato negli esempi di training (default: minimal)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = load_dataset("mteb/banking77")

    id_to_label = {}
    for split in ds.values():
        for row in split:
            id_to_label[row["label"]] = row["label_text"]
    labels = [id_to_label[i] for i in sorted(id_to_label)]
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    LABELS_FILE.write_text(json.dumps(labels, indent=2))

    train = [{"text": r["text"], "label": r["label_text"]} for r in ds["train"]]
    test = [{"text": r["text"], "label": r["label_text"]} for r in ds["test"]]
    random.Random(args.seed).shuffle(train)
    n_valid = int(len(train) * args.valid_ratio)
    valid, train = train[:n_valid], train[n_valid:]

    for name, rows in [("train", train), ("valid", valid), ("test", test)]:
        write_jsonl(RAW_DIR / f"{name}.jsonl", rows)

    def to_chat(row):
        messages = build_messages(row["text"], args.train_prompt, labels)
        return {"messages": messages + [{"role": "assistant", "content": row["label"]}]}

    write_jsonl(LORA_DIR / "train.jsonl", map(to_chat, train))
    write_jsonl(LORA_DIR / "valid.jsonl", map(to_chat, valid))

    table = Table(title="banking77 pronto")
    table.add_column("Split")
    table.add_column("Esempi", justify="right")
    table.add_row("train", str(len(train)))
    table.add_row("valid", str(len(valid)))
    table.add_row("test", str(len(test)))
    table.add_row("etichette", str(len(labels)))
    console.print(table)
    console.print(f"Prompt di training: [bold]{args.train_prompt}[/]")


if __name__ == "__main__":
    main()
