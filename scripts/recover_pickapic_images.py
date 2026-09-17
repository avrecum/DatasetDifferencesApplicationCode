"""Extract only requested original image IDs from a verified author archive.

Parquet image columns require downloading containing shards. Those pinned shards
remain in HF_HOME for resume; only requested image files are extracted. No labels
from the image archive are used.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path

import pyarrow.parquet as pq

from preference_diff.io import file_hash, load_manifest, read_json, write_json
from preference_diff.sources import default_cache, hub_file, validate_image
from preference_diff.recovery import source_uid, verify_identity


def extract_shard(source, filename, requested, output, cache):
    path = hub_file(source, filename, cache)
    parquet = pq.ParquetFile(path)
    columns = [
        k
        for k in (
            "url",
            "prompt",
            "negative_prompt",
            "model_id",
            "image_id",
            "seed",
            "idx",
            "image",
        )
        if k in parquet.schema_arrow.names
    ]
    rows, errors = [], []
    # Iterate record batches to avoid retaining a full archive's decoded images.
    for batch in parquet.iter_batches(batch_size=32, columns=columns):
        for row in batch.to_pylist():
            uid = source_uid(row.get("url"))
            if uid not in requested:
                continue
            target = output / f"{uid}.image"
            tmp = target.with_suffix(".part")
            try:
                verify_identity(row, requested[uid])
                payload = row["image"]["bytes"]
                if not payload or len(payload) > 50 * 1024 * 1024:
                    raise ValueError("Missing or oversized embedded image")
                if target.exists():
                    import hashlib

                    if file_hash(target) != hashlib.sha256(payload).hexdigest():
                        raise ValueError("Existing recovered image has different bytes")
                else:
                    tmp.write_bytes(payload)
                    validate_image(tmp)
                    os.replace(tmp, target)
                audit = validate_image(target)
                rows.append(
                    dict(
                        image_id=requested[uid]["image_id"],
                        image_uid=uid,
                        local_path=str(target.resolve()),
                        source_repository=source["repo"],
                        source_revision=source["revision"],
                        source_file=filename,
                        original_url=row["url"],
                        **audit,
                    )
                )
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                errors.append(dict(image_uid=uid, error=f"{type(exc).__name__}: {exc}"))
    return dict(
        file=filename,
        sha256=file_hash(path),
        bytes=path.stat().st_size,
        extracted=rows,
        errors=errors,
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--coverage", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hf-home", type=Path, default=default_cache())
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    coverage = read_json(args.coverage)
    source = dict(
        repo=coverage["repository"], revision=coverage["revision"], type="dataset"
    )
    manifest = load_manifest(args.manifest)
    required = {
        i
        for e in manifest["events"]
        if e["sampled"] and not e["exclusion_reason"]
        for i in e["candidate_ids"]
    }
    images = {
        i["image_id"].removeprefix("pickapic:"): i
        for i in manifest["images"]
        if i["image_id"] in required
    }
    by_file = {}
    for location in coverage["matched_locations"]:
        if location["uid"] in images:
            by_file.setdefault(location["file"], {})[location["uid"]] = images[
                location["uid"]
            ]
    if set(images) != {uid for values in by_file.values() for uid in values}:
        raise ValueError("Coverage index does not resolve all selected candidate IDs")
    args.image_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        jobs = {
            pool.submit(extract_shard, source, f, req, args.image_dir, args.hf_home): f
            for f, req in sorted(by_file.items())
        }
        for job in as_completed(jobs):
            filename = jobs[job]
            try:
                result = job.result()
            except Exception as exc:
                result = dict(
                    file=filename,
                    extracted=[],
                    errors=[dict(error=f"{type(exc).__name__}: {exc}")],
                )
            results.append(result)
            write_json(
                args.image_dir / "retrieval_provenance.json",
                dict(
                    source=source,
                    images={
                        row["image_uid"]: row
                        for item in results
                        for row in item["extracted"]
                    },
                ),
            )
            write_json(
                args.output,
                dict(
                    source=source,
                    required_images=len(required),
                    resolved_images=sum(len(r["extracted"]) for r in results),
                    shards=results,
                    policy="Original metadata identities/prompts retained; only complete events; image archive contributes bytes, not preference labels",
                ),
            )
            print(
                f"{len(results)}/{len(by_file)} shards; extracted {sum(len(r['extracted']) for r in results)}/{len(required)} images",
                flush=True,
            )


if __name__ == "__main__":
    main()
