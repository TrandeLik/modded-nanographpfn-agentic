"""Strict JSON checkpoints, including partial results if a run fails."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, report: dict) -> None:
    # Serialize first, so invalid/nonfinite values cannot damage a checkpoint.
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
            temporary = Path(f.name)
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
