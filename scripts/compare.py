import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from common import RESULTS_DIR

console = Console()

# (key, label, format, higher is better)
METRICS = [
    ("format_ok", "Formattate correttamente", "pct", True),
    ("accuracy", "Corrette", "pct", True),
    ("prompt_tokens", "Token di prompt (media)", "num", False),
    ("generation_tokens", "Token generati (media)", "num", False),
    ("total_tokens", "Token totali (media)", "num", False),
    ("latency_s", "Latenza (ms)", "ms", False),
]


def adapter_name(run: dict) -> str:
    return Path(run["adapter"]).name if run["adapter"] else "-"


def fmt(value: float, kind: str) -> str:
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "ms":
        return f"{value * 1000:.0f}"
    return f"{value:.1f}"


def fmt_delta(delta: float, kind: str, higher_is_better: bool) -> str:
    if kind == "pct":
        text = f"{delta * 100:+.1f} pt"
    elif kind == "ms":
        text = f"{delta * 1000:+.0f}"
    else:
        text = f"{delta:+.1f}"
    if abs(delta) < 1e-9:
        return f"[dim]{text}[/]"
    good = (delta > 0) == higher_is_better
    return f"[{'green' if good else 'red'}]{text}[/]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Confronta i riepiloghi in results/; le colonne Δ sono la differenza dal primo.")
    parser.add_argument("names", nargs="*", help="scenari da confrontare (default: tutti)")
    args = parser.parse_args()

    if args.names:
        files = [RESULTS_DIR / f"{n}.summary.json" for n in args.names]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"Risultati mancanti: {', '.join(missing)}")
    else:
        files = sorted(RESULTS_DIR.glob("*.summary.json"), key=lambda f: f.stat().st_mtime)
    if not files:
        raise SystemExit("Nessun risultato in results/: lancia prima scripts/evaluate.py.")

    runs = [json.loads(f.read_text()) for f in files]
    if len({r["n"] for r in runs}) > 1:
        console.print("[yellow]Attenzione: gli scenari hanno un numero di esempi diverso, il confronto non è alla pari.[/]")

    table = Table(title="banking77: prima e dopo il fine-tuning")
    table.add_column("Metrica")
    for i, run in enumerate(runs):
        table.add_column(run["name"], justify="right")
        if i > 0:
            table.add_column("Δ", justify="right")

    table.add_row("Prompt", *[x for i, r in enumerate(runs) for x in ([r["prompt"]] if i == 0 else [r["prompt"], ""])])
    table.add_row("Adapter", *[x for i, r in enumerate(runs) for x in ([adapter_name(r)] if i == 0 else [adapter_name(r), ""])])
    table.add_row("Esempi", *[x for i, r in enumerate(runs) for x in ([str(r["n"])] if i == 0 else [str(r["n"]), ""])], end_section=True)

    base = runs[0]
    for key, label, kind, higher in METRICS:
        cells = []
        for i, run in enumerate(runs):
            cells.append(fmt(run[key], kind))
            if i > 0:
                cells.append(fmt_delta(run[key] - base[key], kind, higher))
        table.add_row(label, *cells)

    console.print(table)


if __name__ == "__main__":
    main()
