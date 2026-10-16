#!/usr/bin/env bash
# Resolve the project directory and launch nanoGraphPFN pretraining.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec uv run python train_nano.py --name nanographpfn "$@"
