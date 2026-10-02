import json
import re
import socket
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "results"

OMLX_SETTINGS = Path.home() / ".omlx/settings.json"
OMLX_DEFAULT_MODELS = Path.home() / ".omlx/models"

PROMPT_TYPES = ("minimal", "full")
TASKS = ("classification",)

DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass
class Dataset:
    name: str
    dir: Path
    manifest: dict
    labels: list[str]
    prompts: dict[str, str]

    def split(self, name: str) -> list[dict]:
        return read_jsonl(self.dir / "raw" / f"{name}.jsonl")

    def system_prompt(self, kind: str) -> str:
        if kind not in PROMPT_TYPES:
            raise ValueError(f"prompt sconosciuto: {kind}")
        return self.prompts[kind].replace("{labels}", "\n".join(self.labels))


def dataset_dir(name: str) -> Path:
    if not DATASET_NAME_RE.match(name):
        raise SystemExit(f"Nome di dataset non valido: {name!r} (ammessi lettere, cifre, '.', '_', '-').")
    return DATASETS_DIR / name


def load_dataset_dir(name: str) -> Dataset:
    d = dataset_dir(name)
    manifest_file = d / "manifest.json"
    if not manifest_file.exists():
        available = sorted(p.parent.name for p in DATASETS_DIR.glob("*/manifest.json"))
        hint = f" Disponibili: {', '.join(available)}." if available else ""
        raise SystemExit(f"Dataset {name!r} non trovato in {DATASETS_DIR}: lancia prima ./download.sh.{hint}")
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("task") not in TASKS:
        raise SystemExit(f"Compito {manifest.get('task')!r} non supportato (supportati: {', '.join(TASKS)}).")
    labels = json.loads((d / "labels.json").read_text())
    prompts = yaml.safe_load((d / "prompts.yaml").read_text()) or {}
    missing = [k for k in PROMPT_TYPES if not isinstance(prompts.get(k), str)]
    if missing:
        raise SystemExit(f"{d / 'prompts.yaml'}: mancano i prompt {', '.join(missing)}.")
    return Dataset(name=name, dir=d, manifest=manifest, labels=labels, prompts=prompts)


def build_messages(text: str, system: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def normalize(answer: str) -> str:
    # Deliberately minimal: "card arrival" must not match "card_arrival", since the
    # format_ok metric measures whether the model follows the label format.
    return answer.strip().strip("`\"'").strip().rstrip(".").strip().lower()


def omlx_settings() -> dict:
    try:
        return json.loads(OMLX_SETTINGS.read_text())
    except (OSError, ValueError):
        return {}


def omlx_model_dirs(settings: dict | None = None) -> list[Path]:
    settings = omlx_settings() if settings is None else settings
    dirs = settings.get("model", {}).get("model_dirs") or [str(OMLX_DEFAULT_MODELS)]
    return [Path(d).expanduser() for d in dirs]


def find_models(model_dirs: list[Path], max_depth: int = 3) -> dict[str, Path]:
    """Relative name (e.g. "mlx-community/Qwen2.5-0.5B-Instruct-4bit") -> model folder.

    Like oMLX, any folder containing config.json is a model.
    """
    found: dict[str, Path] = {}

    def walk(base: Path, d: Path, depth: int) -> None:
        if (d / "config.json").is_file() and d != base:
            found.setdefault(d.relative_to(base).as_posix(), d)
            return
        if depth == 0:
            return
        for child in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")):
            walk(base, child, depth - 1)

    for base in model_dirs:
        if base.is_dir():
            walk(base, base, max_depth)
    return found


def resolve_model(name: str, model_dirs: list[Path] | None = None) -> Path:
    """Accepts a model folder, the full relative name, or the leaf folder name (the id oMLX exposes)."""
    path = Path(name).expanduser()
    if (path / "config.json").is_file():
        return path.resolve()

    model_dirs = omlx_model_dirs() if model_dirs is None else model_dirs
    models = find_models(model_dirs)
    if name in models:
        return models[name]
    matches = [p for rel, p in models.items() if rel.split("/")[-1] == name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        options = "\n  ".join(rel for rel, p in models.items() if p in matches)
        raise SystemExit(f"Modello {name!r} ambiguo, usa il nome completo:\n  {options}")
    where = ", ".join(str(d) for d in model_dirs)
    listing = "\n  ".join(sorted(models)) or "(nessuno)"
    raise SystemExit(f"Modello {name!r} non trovato in {where}. Modelli installati:\n  {listing}")


def omlx_running(settings: dict | None = None) -> bool:
    settings = omlx_settings() if settings is None else settings
    server = settings.get("server", {})
    host, port = server.get("host", "127.0.0.1"), int(server.get("port", 8000))
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def warn_if_omlx_running(console) -> None:
    if omlx_running():
        console.print("[yellow]Il server oMLX è attivo: se ha già caricato un modello, la memoria "
                      "si divide fra le due copie. Con i modelli grandi conviene fermarlo (omlx stop).[/]")


ADAPTER_MANIFEST = "manifest.json"


def read_adapter_manifest(adapter: Path) -> dict | None:
    f = adapter / ADAPTER_MANIFEST
    return json.loads(f.read_text()) if f.exists() else None


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
