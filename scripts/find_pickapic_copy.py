"""Find original four-candidate image UIDs in pinned public parquet archives.

Reads UID/URL columns through bounded HTTP ranges, not image bytes or preference
labels. Persistent per-shard metadata enables resuming a bounded search.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import requests

from preference_diff.io import digest, read_json, read_rows, write_json
from preference_diff.sources import HTTPRangeReader
from preference_diff.recovery import source_uid


def inspect_shard(repo, revision, file, cache_dir):
    cache = Path(cache_dir) / digest([repo, revision, file])
    if cache.exists():
        return read_json(cache)
    reader = HTTPRangeReader(
        f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{file['rfilename']}",
        file["size"],
        block_size=65536,
        retries=1,
    )
    parquet = pq.ParquetFile(reader)
    names = parquet.schema_arrow.names
    columns = [
        name
        for name in ("image_uid", "image_0_uid", "image_1_uid", "url")
        if name in names
    ]
    if not columns:
        raise ValueError(f"No verifiable UID or original URL columns in {names}")
    rows = parquet.read(columns=columns).to_pylist()
    locations = []
    for row, item in enumerate(rows):
        for column, value in item.items():
            uid = source_uid(value)
            if uid:
                locations.append(dict(uid=uid, row=row, column=column))
    result = dict(
        repository=repo,
        revision=revision,
        file=file["rfilename"],
        file_size=file["size"],
        rows=len(rows),
        schema=str(parquet.schema_arrow),
        selected_columns=columns,
        uid_locations=locations,
        bytes_downloaded=reader.bytes_downloaded,
        embedded_image_columns=[n for n in names if n in ("image", "jpg_0", "jpg_1")],
    )
    write_json(cache, result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--available-image-ids",
        type=Path,
        help="Write a cohort declaration for prepare after a complete successful scan",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-shards", type=int, default=0)
    args = parser.parse_args()
    endpoint = f"https://huggingface.co/api/datasets/{args.repo}"
    if args.revision:
        endpoint += f"/revision/{args.revision}"
    response = requests.get(endpoint, params={"blobs": "true"}, timeout=30)
    response.raise_for_status()
    repo = response.json()
    revision = repo["sha"]
    if args.revision and revision != args.revision:
        raise ValueError(
            "Resolved revision differs from the requested immutable revision"
        )
    files = sorted(
        (f for f in repo["siblings"] if f["rfilename"].endswith(".parquet")),
        key=lambda f: f["rfilename"],
    )
    total_files = len(files)
    if args.limit_shards:
        files = files[: args.limit_shards]
    original = {
        i["image_id"].removeprefix("pickapic:")
        for i in read_rows(args.manifest / "images.jsonl")
    }
    events = [
        e
        for e in read_rows(args.manifest / "events.jsonl")
        if not e["exclusion_reason"]
    ]
    available, observed, inspected, errors, matches = set(), set(), [], [], []
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    def save():
        complete = [
            e
            for e in events
            if all(i.removeprefix("pickapic:") in available for i in e["candidate_ids"])
        ]
        write_json(
            args.output,
            dict(
                repository=args.repo,
                revision=revision,
                total_archive_shards=total_files,
                inspected_shards=len(inspected),
                failed_shards=errors,
                complete_scan=len(inspected) == total_files and not errors,
                archive_rows=sum(s["rows"] for s in inspected),
                observed_unique_uids=len(observed),
                original_uid_overlap=len(available),
                complete_original_strict_events=len(complete),
                complete_event_ids=[e["event_id"] for e in complete],
                matched_locations=matches,
                shards=inspected,
                bytes_downloaded=sum(s["bytes_downloaded"] for s in inspected),
                interpretation="UID coverage only; no archive preference labels used. A partial or failed scan cannot establish absence from uninspected shards.",
            ),
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        jobs = {
            pool.submit(inspect_shard, args.repo, revision, f, args.cache_dir): f
            for f in files
        }
        for job in as_completed(jobs):
            file = jobs[job]
            try:
                result = job.result()
                for location in result.pop("uid_locations"):
                    observed.add(location["uid"])
                    if location["uid"] in original:
                        available.add(location["uid"])
                        matches.append(dict(file=file["rfilename"], **location))
                inspected.append(result)
                print(
                    f"{len(inspected)}/{len(files)} shards; {len(available)} original UIDs recovered",
                    flush=True,
                )
            except Exception as exc:
                errors.append(
                    dict(file=file["rfilename"], error=f"{type(exc).__name__}: {exc}")
                )
                print(f"Shard failed: {file['rfilename']}: {exc}", flush=True)
            save()
    if not files:
        save()
    if args.available_image_ids:
        if len(inspected) != total_files or errors or not total_files:
            raise ValueError(
                "A cohort declaration requires a complete successful archive scan"
            )
        write_json(
            args.available_image_ids,
            dict(
                image_ids=sorted(f"pickapic:{uid}" for uid in available),
                sources=[
                    dict(
                        repository=args.repo,
                        revision=revision,
                        coverage_audit=str(args.output),
                    )
                ],
            ),
        )


if __name__ == "__main__":
    main()
