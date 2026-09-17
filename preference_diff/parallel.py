"""Small torchrun helpers. Each GPU owns a separate cache; rank zero merges."""

import os
from pathlib import Path
import time

from .io import read_json, write_json


def context():
    import torch
    import torch.distributed as dist

    rank, size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if size > 1 and not dist.is_initialized():
        if torch.cuda.device_count() < size:
            raise RuntimeError("One visible GPU per torchrun worker is required")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("gloo")
    return rank, size


def barrier():
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def partition_images(images, workers):
    """Deterministic balanced partition; aliases of identical content stay together."""
    if workers < 1:
        raise ValueError("Worker count must be positive")
    grouped = {}
    for im in images:
        key = im.get("pixel_hash") or im.get("content_hash") or im["image_id"]
        grouped.setdefault(key, []).append(im)
    parts = [[] for _ in range(workers)]
    for index, key in enumerate(sorted(grouped)):
        parts[index % workers].extend(grouped[key])
    return parts


def score_parallel(images, bank_dir, output, config, cache_root, resume, reuse_scores):
    from .scoring import initialize_score_cache, score_records, open_scores

    rank, workers = context()
    output = Path(output)
    started = time.time()
    parent = None
    if rank == 0:
        parent, _ = initialize_score_cache(
            images, bank_dir, output, config, resume, reuse_scores
        )
    barrier()
    parts = partition_images(images, workers)
    shard = output / "shards" / f"rank-{rank:03d}"
    child = score_records(
        parts[rank], bank_dir, shard, config, cache_root, resume, [output]
    )
    write_json(
        shard / "worker_execution.json",
        dict(
            rank=rank,
            workers=workers,
            device_index=int(os.environ.get("LOCAL_RANK", rank)),
            assigned_unique_images=len(child.images),
            valid_images=int(child.valid.sum()),
            elapsed_seconds=time.time() - started,
        ),
    )
    barrier()
    if rank == 0:
        audits = []
        for index in range(workers):
            path = output / "shards" / f"rank-{index:03d}"
            parent.reuse_from(path)
            _, mapping, valid, _, _, state = open_scores(path)
            for iid, row in mapping.items():
                target = parent.row_index[iid]
                if not valid[row] and not parent.valid[target]:
                    parent.fail(
                        target, state["errors"].get(str(row), "uncommitted shard row")
                    )
            audits.append(read_json(path / "worker_execution.json"))
        write_json(
            output / "distributed_execution.json",
            dict(
                workers=audits,
                policy="content-group round-robin; disjoint per-GPU caches; rank-zero atomic merge",
                elapsed_seconds=time.time() - started,
            ),
        )
        # Preserve actual per-worker reference audits and one common bank diagnostic.
        write_json(
            output / "reference_validation.json",
            dict(
                workers=[
                    read_json(
                        output
                        / "shards"
                        / f"rank-{i:03d}"
                        / "reference_validation.json"
                    )
                    for i in range(workers)
                    if (
                        output
                        / "shards"
                        / f"rank-{i:03d}"
                        / "reference_validation.json"
                    ).exists()
                ]
            ),
        )
        for index in range(workers):
            audit = output / "shards" / f"rank-{index:03d}" / "text_mode_audit.json"
            if audit.exists():
                write_json(output / "text_mode_audit.json", read_json(audit))
                break
    barrier()
    return parent
