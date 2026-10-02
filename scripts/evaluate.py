"""Valuta un modello installato in oMLX, base o con adapter LoRA, sul test set di un dataset.

Uso:
    ./evaluate.sh --model Qwen2.5-0.5B-Instruct-4bit --dataset-name banking77 \\
                  --prompt-type full --output results/base-full.jsonl
    ./evaluate.sh --model Qwen2.5-0.5B-Instruct-4bit --dataset-name banking77 \\
                  --adapter adapters/qwen05-banking77 \\
                  --prompt-type minimal --output results/ft-minimal.jsonl
    # iterazione veloce su un sottoinsieme: aggiungi --limit 200

Salva:
    <output>                 una riga per esempio (risposta, token, latenza, esito)
    <output>.summary.json    le metriche aggregate (al posto di .jsonl, se l'output finisce così)

Se i file esistono vengono sovrascritti.
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from common import (PROMPT_TYPES, build_messages, load_dataset_dir, normalize, read_adapter_manifest,
                    resolve_model, warn_if_omlx_running, write_jsonl)

console = Console()


def summary_path(output: Path) -> Path:
    stem = output.name[: -len(".jsonl")] if output.name.endswith(".jsonl") else output.name
    return output.with_name(f"{stem}.summary.json")


def run_name(output: Path) -> str:
    return output.name[: -len(".jsonl")] if output.name.endswith(".jsonl") else output.stem


def score(answer: str, expected: str, label_set: set[str]) -> tuple[bool, bool]:
    """(formattata, corretta): la normalizzazione vale per entrambi i lati."""
    pred = normalize(answer)
    return pred in label_set, pred == normalize(expected)


def check_adapter(adapter: Path, model_path: Path, dataset: str) -> None:
    if not (adapter / "adapters.safetensors").exists():
        raise SystemExit(f"{adapter}/adapters.safetensors non trovato: non è la cartella di un adapter.")
    manifest = read_adapter_manifest(adapter)
    if manifest is None:
        console.print(f"[yellow]{adapter} non ha un manifest.json: nessun controllo di compatibilità.[/]")
        return
    if Path(manifest["model_path"]) != model_path:
        raise SystemExit(f"L'adapter è stato addestrato su {manifest['model']} ({manifest['model_path']}), "
                         f"non su {model_path}.")
    if manifest.get("dataset") != dataset:
        console.print(f"[yellow]L'adapter è stato addestrato su {manifest.get('dataset')!r}, "
                      f"lo stai valutando su {dataset!r}.[/]")


def summarize(rows: list[dict], meta: dict) -> dict:
    n = len(rows)
    return {
        **meta,
        "n": n,
        "format_ok": sum(r["format_ok"] for r in rows) / n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows) / n,
        "generation_tokens": sum(r["generation_tokens"] for r in rows) / n,
        "total_tokens": sum(r["prompt_tokens"] + r["generation_tokens"] for r in rows) / n,
        "latency_s": sum(r["latency_s"] for r in rows) / n,
    }


def print_summary(s: dict, rows: list[dict]) -> None:
    table = Table(title=f"Risultati: {s['name']}")
    table.add_column("Metrica")
    table.add_column("Valore", justify="right")
    table.add_row("Esempi", str(s["n"]))
    table.add_row("Risposte formattate correttamente", f"{s['format_ok']:.1%}")
    table.add_row("Risposte corrette", f"{s['accuracy']:.1%}")
    table.add_row("Token di prompt (media)", f"{s['prompt_tokens']:.1f}")
    table.add_row("Token generati (media)", f"{s['generation_tokens']:.1f}")
    table.add_row("Token totali (media)", f"{s['total_tokens']:.1f}")
    table.add_row("Latenza per richiesta", f"{s['latency_s'] * 1000:.0f} ms")
    console.print(table)

    errors = Counter((r["label"], r["answer"][:40]) for r in rows if not r["correct"])
    if errors:
        err = Table(title="Errori più frequenti")
        err.add_column("Atteso")
        err.add_column("Risposta del modello")
        err.add_column("Volte", justify="right")
        for (expected, answer), count in errors.most_common(10):
            err.add_row(expected, repr(answer), str(count))
        console.print(err)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="nome del modello in oMLX (o cartella del modello)")
    parser.add_argument("--dataset-name", required=True, help="nome locale dato a download.sh")
    parser.add_argument("--prompt-type", required=True, choices=PROMPT_TYPES,
                        help="full = system prompt con l'elenco delle etichette; minimal = solo istruzione breve")
    parser.add_argument("--output", required=True, type=Path, help="file .jsonl delle predizioni")
    parser.add_argument("--adapter", type=Path, default=None, help="cartella dell'adapter LoRA (omessa = modello base)")
    parser.add_argument("--split", default="test", choices=["test", "valid"])
    parser.add_argument("--limit", type=int, default=None, help="valuta solo N esempi (campione casuale fisso)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    dataset = load_dataset_dir(args.dataset_name)
    model_path = resolve_model(args.model)
    if args.adapter:
        check_adapter(args.adapter, model_path, dataset.name)

    label_set = {normalize(label) for label in dataset.labels}
    system = dataset.system_prompt(args.prompt_type)
    examples = dataset.split(args.split)
    if args.limit:
        # Gli split possono essere ordinati per etichetta: campioniamo per averle tutte.
        examples = random.Random(args.seed).sample(examples, min(args.limit, len(examples)))

    console.print(f"Modello: [bold]{args.model}[/]  adapter: [bold]{args.adapter or '-'}[/]  "
                  f"dataset: [bold]{dataset.name}[/]  prompt: [bold]{args.prompt_type}[/]")
    warn_if_omlx_running(console)

    from mlx_lm import load, stream_generate  # import lento: dopo la validazione degli argomenti

    model, tokenizer = load(str(model_path), adapter_path=str(args.adapter) if args.adapter else None)

    rows = []
    progress = Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
                        TextColumn("acc {task.fields[acc]}"), TimeRemainingColumn(), console=console)
    with progress:
        task = progress.add_task(run_name(args.output), total=len(examples), acc="-")
        for ex in examples:
            prompt = tokenizer.apply_chat_template(build_messages(ex["text"], system), add_generation_prompt=True)

            start = time.perf_counter()
            answer, last = "", None
            # Nessun sampler: decoding greedy, deterministico.
            for last in stream_generate(model, tokenizer, prompt, max_tokens=args.max_tokens):
                answer += last.text
            latency = time.perf_counter() - start

            format_ok, correct = score(answer, ex["label"], label_set)
            rows.append({
                "text": ex["text"],
                "label": ex["label"],
                "answer": answer,
                "format_ok": format_ok,
                "correct": correct,
                "prompt_tokens": last.prompt_tokens,
                "generation_tokens": last.generation_tokens,
                "latency_s": latency,
            })
            acc = sum(r["correct"] for r in rows) / len(rows)
            progress.update(task, advance=1, acc=f"{acc:.1%}")

    meta = {"name": run_name(args.output), "dataset": dataset.name, "prompt": args.prompt_type,
            "adapter": str(args.adapter) if args.adapter else None, "model": args.model,
            "model_path": str(model_path), "split": args.split, "limit": args.limit}
    summary = summarize(rows, meta)
    write_jsonl(args.output, rows)
    summary_path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print_summary(summary, rows)
    console.print(f"Salvati {args.output} e {summary_path(args.output)}")


if __name__ == "__main__":
    main()
