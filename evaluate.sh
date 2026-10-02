#!/usr/bin/env bash
# Wrapper di scripts/evaluate.py: gli argomenti passano invariati, --help per l'elenco.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec uv run --project "$here" python "$here/scripts/evaluate.py" "$@"
