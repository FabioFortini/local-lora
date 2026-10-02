#!/usr/bin/env bash
# Wrapper di scripts/finetune.py: gli argomenti passano invariati, --help per l'elenco.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec uv run --project "$here" python "$here/scripts/finetune.py" "$@"
