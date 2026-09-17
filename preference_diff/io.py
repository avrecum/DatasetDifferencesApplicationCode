"""Small, atomic, explicit artifact helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def clean(value: Any):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def digest(value):
    return hashlib.sha256(
        json.dumps(
            clean(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, value):
    atomic_text(
        path,
        json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path, rows):
    atomic_text(
        path,
        "".join(
            json.dumps(clean(r), ensure_ascii=False, allow_nan=False) + "\n"
            for r in rows
        ),
    )


def read_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return clean(pq.read_table(path).to_pylist())
    if path.suffix == ".jsonl":
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if path.suffix == ".csv":
        with path.open() as f:
            return list(csv.DictReader(f))
    return read_json(path)


def write_csv(path, rows, fields=None):
    import io

    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    for row in rows:
        w.writerow(
            {
                k: json.dumps(clean(v), ensure_ascii=False)
                if isinstance(v, (dict, list))
                else clean(v)
                for k, v in row.items()
            }
        )
    atomic_text(path, buf.getvalue())


def save_manifest(directory, manifest):
    directory = Path(directory)
    for kind in ("images", "events", "comparisons"):
        write_jsonl(directory / f"{kind}.jsonl", manifest.get(kind, []))
    write_json(directory / "dataset_audit.json", manifest["audit"])
    write_json(directory / "provenance.json", manifest.get("provenance", {}))


def load_manifest(directory):
    directory = Path(directory)
    return {
        **{
            k: read_rows(directory / f"{k}.jsonl")
            for k in ("images", "events", "comparisons")
        },
        "audit": read_json(directory / "dataset_audit.json"),
        "provenance": read_json(directory / "provenance.json"),
    }
