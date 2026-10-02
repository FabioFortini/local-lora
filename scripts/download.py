import argparse
import json
import random
import shutil
from datetime import datetime, timezone

from datasets import ClassLabel, Value, load_dataset
from rich.console import Console
from rich.table import Table

from common import dataset_dir, normalize, write_jsonl

console = Console()

LABEL_NAMES = ("label_text", "label", "labels", "intent", "category", "class", "target")
TEXT_NAMES = ("text", "sentence", "content", "utterance", "query", "question", "review", "document")
VALID_SPLITS = ("validation", "valid", "dev")
MANY_LABELS = 200

DEFAULT_PROMPTS = """\
# {labels} is replaced with the label list, one per line. download.sh keeps edits unless --overwrite.
minimal: >-
  Classify the text into its label.
  Answer with the label only.
full: |-
  Classify the text into exactly one of the following labels:
  {labels}

  Answer with the label only, exactly as written above, with no other text.
"""


def is_string(feature) -> bool:
    return isinstance(feature, Value) and feature.dtype in ("string", "large_string")


def is_int(feature) -> bool:
    return isinstance(feature, Value) and feature.dtype.startswith(("int", "uint"))


def pick_label_column(features, override: str | None = None) -> str:
    if override:
        if override not in features:
            raise SystemExit(f"Colonna {override!r} assente. Colonne: {', '.join(features)}")
        if not (isinstance(features[override], ClassLabel) or is_string(features[override])):
            raise SystemExit(f"La colonna {override!r} deve contenere testo o una ClassLabel con i nomi "
                             f"delle etichette, non {features[override]}.")
        return override

    class_labels = [c for c, f in features.items() if isinstance(f, ClassLabel)]
    if len(class_labels) == 1:
        return class_labels[0]
    if len(class_labels) > 1:
        for name in LABEL_NAMES:
            if name in class_labels:
                return name
        raise SystemExit(f"Più colonne di etichette ({', '.join(class_labels)}): scegli con --label-column.")

    for name in LABEL_NAMES:
        if name in features and is_string(features[name]):
            return name
    raise SystemExit("Nessuna colonna di etichette riconosciuta: indicala con --label-column. "
                     f"Colonne: {', '.join(f'{c} ({f})' for c, f in features.items())}")


def pick_text_column(features, label_column: str, override: str | None = None) -> str:
    if override:
        if override not in features or not is_string(features[override]):
            raise SystemExit(f"La colonna di testo {override!r} deve esistere e contenere stringhe.")
        if override == label_column:
            raise SystemExit("Testo ed etichetta non possono essere la stessa colonna.")
        return override

    strings = [c for c, f in features.items() if is_string(f) and c != label_column]
    for name in TEXT_NAMES:
        if name in strings:
            return name
    if len(strings) == 1:
        return strings[0]
    raise SystemExit("Colonna di testo ambigua o assente: indicala con --text-column. "
                     f"Colonne di testo: {', '.join(strings) or 'nessuna'}")


def label_order(features, label_column: str, splits: dict) -> list[str] | None:
    """Label order defined by the dataset, or None (caller sorts alphabetically).

    A ClassLabel carries names in id order; a "<x>_text" string column paired with an
    integer "<x>" column (as in mteb/banking77) is ordered by that id.
    """
    feature = features[label_column]
    if isinstance(feature, ClassLabel):
        return list(feature.names)
    if label_column.endswith("_text"):
        id_column = label_column[: -len("_text")]
        if id_column in features and is_int(features[id_column]):
            id_to_label = {}
            for split in splits.values():
                for i, label in zip(split[id_column], split[label_column]):
                    if id_to_label.setdefault(i, label) != label:
                        return None
            if len(set(id_to_label.values())) == len(id_to_label):
                return [id_to_label[i] for i in sorted(id_to_label)]
    return None


def to_rows(split, text_column: str, label_column: str) -> tuple[list[dict], int]:
    """Rows without text or label are dropped (e.g. the unlabeled GLUE test split)."""
    feature = split.features[label_column]
    names = feature.names if isinstance(feature, ClassLabel) else None
    rows, dropped = [], 0
    for text, label in zip(split[text_column], split[label_column]):
        if names is not None:
            label = names[label] if label is not None and 0 <= label < len(names) else None
        if not text or not str(text).strip() or label is None or not str(label).strip():
            dropped += 1
            continue
        rows.append({"text": text, "label": label})
    return rows, dropped


def assign_splits(train: list, valid: list | None, test: list | None,
                  valid_ratio: float, test_ratio: float, seed: int):
    """Missing splits are carved from the shuffled train: test first, then validation."""
    train = list(train)
    random.Random(seed).shuffle(train)
    origin = {"train": "train", "valid": "valid", "test": "test"}
    if not test:
        n = int(len(train) * test_ratio)
        test, train = train[:n], train[n:]
        origin["test"] = "train"
    if not valid:
        n = int(len(train) * valid_ratio)
        valid, train = train[:n], train[n:]
        origin["valid"] = "train"
    if not train or not valid or not test:
        raise SystemExit(f"Split vuoti dopo la suddivisione (train {len(train)}, valid {len(valid)}, "
                         f"test {len(test)}): il dataset è troppo piccolo o i rapporti troppo bassi.")
    return train, valid, test, origin


def main() -> None:
    parser = argparse.ArgumentParser(description="Scarica un dataset di classificazione da Hugging Face in datasets/<name>/.")
    parser.add_argument("--source", required=True, help="id del dataset su Hugging Face, es. mteb/banking77")
    parser.add_argument("--name", required=True, help="nome locale, usato poi come --dataset-name")
    parser.add_argument("--subset", default=None, help="configurazione del dataset, es. sst2 per nyu-mll/glue")
    parser.add_argument("--revision", default=None, help="branch, tag o commit del dataset")
    parser.add_argument("--text-column", default=None, help="colonna del testo (default: riconosciuta)")
    parser.add_argument("--label-column", default=None, help="colonna dell'etichetta (default: riconosciuta)")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="quota del train usata come validation, se manca")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="quota del train usata come test, se manca")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="sostituisce un dataset con lo stesso nome ma sorgente diversa")
    args = parser.parse_args()

    out = dataset_dir(args.name)
    manifest_file = out / "manifest.json"
    if manifest_file.exists() and not args.overwrite:
        old = json.loads(manifest_file.read_text())
        if (old.get("source"), old.get("subset")) != (args.source, args.subset):
            raise SystemExit(f"datasets/{args.name} contiene già {old.get('source')}"
                             f"{'/' + old['subset'] if old.get('subset') else ''}: "
                             "scegli un altro --name o aggiungi --overwrite.")

    try:
        ds = load_dataset(args.source, args.subset, revision=args.revision)
    except Exception as e:  # datasets raises unrelated types for network, name, config and auth errors
        raise SystemExit(f"Download di {args.source} fallito: {e}")

    if "train" not in ds:
        raise SystemExit(f"Il dataset non ha uno split 'train' (split: {', '.join(ds)}).")
    features = ds["train"].features
    label_column = pick_label_column(features, args.label_column)
    text_column = pick_text_column(features, label_column, args.text_column)

    mapping = Table(title=f"{args.source}: colonne scelte")
    mapping.add_column("Ruolo")
    mapping.add_column("Colonna")
    mapping.add_column("Tipo")
    mapping.add_row("testo", text_column, str(features[text_column]))
    mapping.add_row("etichetta", label_column, str(features[label_column]))
    console.print(mapping)

    valid_name = next((s for s in VALID_SPLITS if s in ds), None)
    raw, dropped = {}, {}
    for role, hf_name in [("train", "train"), ("valid", valid_name), ("test", "test")]:
        if hf_name in ds:
            raw[role], dropped[role] = to_rows(ds[hf_name], text_column, label_column)
            if dropped[role]:
                console.print(f"[yellow]{hf_name}: scartate {dropped[role]} righe senza testo o etichetta.[/]")
            if not raw[role] and role != "train":
                console.print(f"[yellow]{hf_name}: nessuna riga etichettata, lo split viene ricavato dal train.[/]")
    train, valid, test, origin = assign_splits(raw["train"], raw.get("valid"), raw.get("test"),
                                               args.valid_ratio, args.test_ratio, args.seed)

    used = {r["label"] for rows in (train, valid, test) for r in rows}
    order = label_order(features, label_column, {s: ds[s] for s in ds})
    labels = [label for label in order if label in used] if order else sorted(used)
    normalized = {}
    for label in labels:
        if normalize(label) in normalized:
            raise SystemExit(f"Le etichette {normalized[normalize(label)]!r} e {label!r} coincidono dopo la "
                             "normalizzazione: la valutazione non saprebbe distinguerle.")
        normalized[normalize(label)] = label
    unseen = {r["label"] for r in test} - {r["label"] for r in train}
    if unseen:
        console.print(f"[yellow]{len(unseen)} etichette compaiono nel test ma non nel train.[/]")
    if len(labels) > MANY_LABELS:
        console.print(f"[yellow]{len(labels)} etichette: il prompt full sarà molto lungo, preferisci minimal.[/]")
    if len(labels) < 2:
        raise SystemExit("Serve almeno due etichette per un compito di classificazione.")

    if args.overwrite and out.exists():
        shutil.rmtree(out / "raw", ignore_errors=True)
    for name, rows in [("train", train), ("valid", valid), ("test", test)]:
        write_jsonl(out / "raw" / f"{name}.jsonl", rows)
    (out / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False))
    prompts_file = out / "prompts.yaml"
    kept_prompts = prompts_file.exists() and not args.overwrite
    if not kept_prompts:
        prompts_file.write_text(DEFAULT_PROMPTS)

    manifest = {
        "name": args.name,
        "task": "classification",
        "source": args.source,
        "subset": args.subset,
        "revision": args.revision,
        "text_column": text_column,
        "label_column": label_column,
        "splits": {
            "train": {"n": len(train), "from": "train"},
            "valid": {"n": len(valid), "from": valid_name if origin["valid"] == "valid" else "train"},
            "test": {"n": len(test), "from": "test" if origin["test"] == "test" else "train"},
        },
        "dropped": dropped,
        "num_labels": len(labels),
        "valid_ratio": args.valid_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    table = Table(title=f"{args.name} pronto in datasets/{args.name}")
    table.add_column("Split")
    table.add_column("Esempi", justify="right")
    table.add_column("Origine")
    for split in ("train", "valid", "test"):
        info = manifest["splits"][split]
        table.add_row(split, str(info["n"]), info["from"])
    table.add_row("etichette", str(len(labels)), "")
    console.print(table)
    console.print(f"Prompt: {prompts_file} " + ("[dim](esistente, non modificato)[/]" if kept_prompts else "(da personalizzare)"))


if __name__ == "__main__":
    main()
