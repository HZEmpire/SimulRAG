"""Dataset, cache, and run-artifact helpers."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

BUILTIN_DATASETS = {
    "climate": DATA_ROOT / "climate" / "questions.json",
    "epidemiology": DATA_ROOT / "epidemiology" / "questions.json",
    "urban": DATA_ROOT / "urban" / "questions.json",
}

DATASET_ALIASES = {
    "climate": "climate",
    "clim": "climate",
    "c": "climate",
    "epidemiology": "epidemiology",
    "epi": "epidemiology",
    "e": "epidemiology",
    "urban": "urban",
    "sumo": "urban",
    "u": "urban",
}


@dataclass(slots=True)
class Dataset:
    name: str
    path: Path
    records: list[dict[str, str]]
    builtin: bool


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _resolve_custom_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    data_relative = (DATA_ROOT / path).resolve()
    if data_relative.exists():
        return data_relative
    return (PROJECT_ROOT / path).resolve()


def load_dataset(dataset: str, custom_path: str | None = None) -> Dataset:
    """Load a built-in alias or a user-supplied JSON dataset."""
    normalized = DATASET_ALIASES.get(dataset.lower())
    if normalized and custom_path is None:
        path = BUILTIN_DATASETS[normalized]
        builtin = True
        name = normalized
    else:
        source = custom_path or dataset
        path = _resolve_custom_path(source)
        builtin = False
        name = slugify(path.stem)

    raw = _read_json(path)
    if not isinstance(raw, list):
        raise TypeError(f"Dataset must be a JSON array: {path}")

    records: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(f"Dataset item {index} must be an object")
        # Legacy open_question is accepted only when importing old data.
        question = item.get("question") or item.get("open_question")
        reference = item.get("reference_answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Dataset item {index} has no non-empty question")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"Dataset item {index} has no non-empty reference_answer")
        records.append(
            {
                "question": question.strip(),
                "reference_answer": reference.strip(),
            }
        )

    return Dataset(name=name, path=path, records=records, builtin=builtin)


def load_builtin_evidence(dataset_name: str) -> dict[str, dict[str, str]]:
    path = DATA_ROOT / "cache" / dataset_name / "simulator_evidence.json"
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise TypeError(f"Simulator cache must be a JSON array: {path}")
    evidence: dict[str, dict[str, str]] = {}
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("question"), str):
            raise TypeError(f"Malformed simulator cache item in {path}")
        evidence[item["question"]] = {
            "derived_quantitative_question": str(
                item.get("derived_quantitative_question", "")
            ),
            "derived_quantitative_answer": str(
                item.get("derived_quantitative_answer", "")
            ),
        }
    return evidence


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return slug.lower() or "dataset"


def create_run_dir(dataset_name: str, run_name: str | None = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = slugify(run_name) if run_name else stamp
    base = DATA_ROOT / "runs" / slugify(dataset_name)
    path = base / label
    suffix = 1
    while path.exists():
        path = base / f"{label}-{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def latest_run(dataset_name: str | None = None) -> Path:
    root = DATA_ROOT / "runs"
    if dataset_name:
        root = root / slugify(DATASET_ALIASES.get(dataset_name, dataset_name))
    candidates = [path for path in root.glob("**/06_final_output.json")]
    if not candidates:
        raise FileNotFoundError(f"No completed run found below {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def write_json(path: Path, value: Any) -> None:
    """Atomically write a UTF-8 JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
