"""Dataset/model discovery without scientific Python dependencies."""

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPHLAND_NAMES = (
    "artnet-exp", "artnet-views", "avazu-ctr", "city-reviews", "city-roads-L", "city-roads-M",
    "hm-categories", "hm-prices", "pokec-regions", "tolokers-2", "twitch-views",
    "web-fraud", "web-topics", "web-traffic",
)
MODELS = {
    "rf50": {"n_estimators": 50, "nfa": False},
    "rf500": {"n_estimators": 500, "nfa": False},
    "rf50_nfa": {"n_estimators": 50, "nfa": True},
    "rf500_nfa": {"n_estimators": 500, "nfa": True},
}
FILES = ("info.yaml", "features.csv", "targets.csv", "edgelist.csv", "split_masks_RL.csv")


def dataset_identity(name: str) -> tuple[str, int | None]:
    match = re.fullmatch(r"(.+)-t-(\d+)", name)
    base, index = (match[1], int(match[2])) if match else (name, None)
    if base not in GRAPHLAND_NAMES:
        raise ValueError(f"Not a GraphLand dataset: {name}")
    return base, index


def discover_datasets(root: Path, names: list[str] | None = None, variants: str = "all") -> list[Path]:
    if variants not in {"all", "full", "tiny"}:
        raise ValueError(f"Unknown variants: {variants}")
    if not root.is_dir():
        raise ValueError(f"Data directory does not exist: {root}")
    requested = set(names or [])
    if names and len(requested) != len(names):
        raise ValueError("Duplicate dataset selectors")
    for name in requested:
        dataset_identity(name)
    found, matched = [], set()
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        try:
            base, index = dataset_identity(directory.name)
        except ValueError:
            continue
        if variants == "full" and index is not None or variants == "tiny" and index is None:
            continue
        selectors = requested.intersection({base, directory.name})
        if requested and not selectors:
            continue
        matched.update(selectors)
        for filename in FILES:
            if not (directory / filename).is_file():
                raise ValueError(f"Missing dataset file: {directory / filename}")
        found.append(directory)
    if requested - matched:
        raise ValueError(f"No matching {variants} datasets for: {', '.join(sorted(requested - matched))}")
    if not found:
        raise ValueError(f"No GraphLand datasets found in {root}")
    return sorted(found, key=lambda p: (dataset_identity(p.name)[0],
                                       -1 if dataset_identity(p.name)[1] is None else dataset_identity(p.name)[1]))
