"""Percorsi, prompt e utilità condivise dagli script dell'esperimento banking77."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # split con testo ed etichetta, usati dalla valutazione
LORA_DIR = DATA_DIR / "lora"  # split in formato chat, usati da mlx_lm lora
LABELS_FILE = DATA_DIR / "labels.json"
RESULTS_DIR = ROOT / "results"
ADAPTERS_DIR = ROOT / "adapters"

_OMLX_MODEL = Path.home() / ".omlx/models/mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_MODEL = str(_OMLX_MODEL) if _OMLX_MODEL.exists() else "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

SYSTEM_MINIMAL = (
    "Classify the banking customer message into its intent label. "
    "Answer with the label only."
)

SYSTEM_FULL = (
    "You are a banking customer support classifier. "
    "Classify the customer message into exactly one of the following intent labels:\n"
    "{labels}\n\n"
    "Answer with the label only, exactly as written above, with no other text."
)


def load_labels() -> list[str]:
    if not LABELS_FILE.exists():
        raise SystemExit(f"{LABELS_FILE} non trovato: lancia prima `uv run scripts/download.py`.")
    return json.loads(LABELS_FILE.read_text())


def system_prompt(kind: str, labels: list[str]) -> str:
    if kind == "minimal":
        return SYSTEM_MINIMAL
    if kind == "full":
        return SYSTEM_FULL.format(labels="\n".join(labels))
    raise ValueError(f"prompt sconosciuto: {kind}")


def build_messages(text: str, kind: str, labels: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt(kind, labels)},
        {"role": "user", "content": text},
    ]


def normalize(answer: str) -> str:
    """Pulizia minima: spazi, virgolette, backtick e punto finale, poi minuscolo.

    Non corregge altro: una risposta come "card arrival" (senza underscore) resta
    non formattata, perché l'obiettivo è misurare se il modello rispetta il formato.
    """
    return answer.strip().strip("`\"'").strip().rstrip(".").strip().lower()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
