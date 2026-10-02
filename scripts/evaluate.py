"""Valuta il modello (base o con adapter LoRA) sul test set di banking77.

Uso:
    uv run scripts/evaluate.py --name base-full --prompt full
    uv run scripts/evaluate.py --name ft-full --prompt full --adapter adapters/banking77
    uv run scripts/evaluate.py --name ft-minimal --prompt minimal --adapter adapters/banking77
    # iterazione veloce su un sottoinsieme:
    uv run scripts/evaluate.py --name base-full --prompt full --limit 200

Salva:
    results/<name>.jsonl         una riga per esempio (risposta, token, latenza, esito)
    results/<name>.summary.json  le metriche aggregate lette da compare.py
"""

import argparse
import json
import random
import time
from collections import Counter

from mlx_lm import load, stream_generate
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from common import DEFAULT_MODEL, RAW_DIR, RESULTS_DIR, build_messages, load_labels, normalize, read_jsonl, write_jsonl

console = Console()


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
    parser.add_argument("--name", required=True, help="nome dello scenario, usato per i file in results/")
    parser.add_argument("--prompt", choices=["full", "minimal"], default="full",
                        help="full = system prompt con le 77 etichette; minimal = solo istruzione breve")
    parser.add_argument("--adapter", default=None, help="cartella dell'adapter LoRA (omessa = modello base)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="test", choices=["test", "valid"])
    parser.add_argument("--limit", type=int, default=None, help="valuta solo N esempi (campione casuale fisso)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    labels = load_labels()
    label_set = set(labels)
    examples = read_jsonl(RAW_DIR / f"{args.split}.jsonl")
    if args.limit:
        # Il test è ordinato per etichetta: campioniamo per avere tutte le classi.
        examples = random.Random(args.seed).sample(examples, min(args.limit, len(examples)))

    console.print(f"Modello: [bold]{args.model}[/]  adapter: [bold]{args.adapter or '-'}[/]  prompt: [bold]{args.prompt}[/]")
    model, tokenizer = load(args.model, adapter_path=args.adapter)

    rows = []
    progress = Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
                        TextColumn("acc {task.fields[acc]}"), TimeRemainingColumn(), console=console)
    with progress:
        task = progress.add_task(args.name, total=len(examples), acc="-")
        for ex in examples:
            messages = build_messages(ex["text"], args.prompt, labels)
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

            start = time.perf_counter()
            answer, last = "", None
            # Nessun sampler: decoding greedy, deterministico.
            for last in stream_generate(model, tokenizer, prompt, max_tokens=args.max_tokens):
                answer += last.text
            latency = time.perf_counter() - start

            pred = normalize(answer)
            rows.append({
                "text": ex["text"],
                "label": ex["label"],
                "answer": answer,
                "format_ok": pred in label_set,
                "correct": pred == ex["label"],
                "prompt_tokens": last.prompt_tokens,
                "generation_tokens": last.generation_tokens,
                "latency_s": latency,
            })
            acc = sum(r["correct"] for r in rows) / len(rows)
            progress.update(task, advance=1, acc=f"{acc:.1%}")

    meta = {"name": args.name, "prompt": args.prompt, "adapter": args.adapter,
            "model": args.model, "split": args.split, "limit": args.limit}
    summary = summarize(rows, meta)
    RESULTS_DIR.mkdir(exist_ok=True)
    write_jsonl(RESULTS_DIR / f"{args.name}.jsonl", rows)
    (RESULTS_DIR / f"{args.name}.summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary, rows)


if __name__ == "__main__":
    main()
