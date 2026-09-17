"""Bounded text-only provenance diagnostics; never replace the supplied concept bank.

The candidate settings are fixed before observing their agreement with the bank.
Image preferences and discovery/test outcomes are not inputs to this diagnostic.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np

from .io import write_json, read_json
from .scoring import ScoreConfig, load_model, load_text_embeddings
from .sources import default_cache
from .parallel import context, barrier


TEMPLATES = (
    "{}",
    "a {}",
    "a {}.",
    "a photo of {}",
    "a photo of {}.",
    "a photo of a {}.",
    "a photo of the {}.",
    "an image of {}.",
    "a picture of {}.",
    "{} in the image",
    "there is {} in the image.",
)


def diagnose(bank_dir, output, config, cache_root, sample_size=32):
    import torch
    import torch.nn.functional as F

    if sample_size < 2:
        raise ValueError("Use at least two diagnostic entries")
    rank, workers = context()
    text, vocab, bank_audit = load_text_embeddings(bank_dir, return_audit=True)
    model, _, tokenizer = load_model(config, cache_root)
    indices = np.linspace(0, len(vocab) - 1, min(sample_size, len(vocab)), dtype=int)
    target = text[indices].to(config.device)
    full_bank = text.to(config.device)
    words = [vocab[i] for i in indices]
    records = []
    torch.backends.cuda.matmul.allow_tf32 = False
    setting_index = -1
    with torch.inference_mode():
        for mode, length in (("box", 64), ("short", 64), ("long", 196)):
            for template in TEMPLATES:
                for use_mask in (True, False):
                    setting_index += 1
                    if setting_index % workers != rank:
                        continue
                    rendered = [template.format(word.lower()) for word in words]
                    inputs = tokenizer(
                        rendered,
                        padding="max_length",
                        max_length=length,
                        truncation=True,
                        return_tensors="pt",
                        return_attention_mask=use_mask,
                    ).to(config.device)
                    if not use_mask:
                        inputs.pop("attention_mask", None)
                    if ("attention_mask" in inputs) != use_mask:
                        raise ValueError(
                            "Tokenizer did not honor explicit attention-mask setting"
                        )
                    features = F.normalize(
                        model.get_text_features(**inputs, walk_type=mode).float(),
                        dim=-1,
                    )
                    cosine = (features * target).sum(dim=-1)
                    differences = (features - target).abs().amax(dim=-1)
                    # Identity check over the full original bank can expose row misalignment.
                    neighbors = (features @ full_bank.T).argmax(dim=1)
                    row = dict(
                        walk_type=mode,
                        max_length=length,
                        template=template,
                        attention_mask=use_mask,
                        lowercased=True,
                        mean_cosine=float(cosine.mean()),
                        minimum_cosine=float(cosine.min()),
                        maximum_absolute_error=float(differences.max()),
                        same_row_nearest_fraction=float(
                            (
                                neighbors
                                == torch.as_tensor(indices, device=config.device)
                            )
                            .float()
                            .mean()
                        ),
                        cosine_per_entry=cosine.cpu().tolist(),
                        nearest_bank_ids=neighbors.cpu().tolist(),
                        worker_rank=rank,
                    )
                    records.append(row)
                    print(
                        f"{mode} mask={use_mask} {template!r}: cosine={row['mean_cosine']:.6f}",
                        flush=True,
                    )
    records.sort(key=lambda r: -r["mean_cosine"])
    result = dict(
        model=asdict(config),
        diagnostic_version=2,
        bank_audit=bank_audit,
        indices=indices.tolist(),
        words=words,
        settings=records,
        selection="Evenly spaced vocabulary entries; fixed templates/modes; no preference data used",
        interpretation="Agreement is a diagnostic, not original generation metadata. Original bank remains unchanged.",
    )
    if workers > 1:
        shard = Path(output).parent / "bank_diagnostic_shards" / f"rank-{rank:03d}.json"
        write_json(shard, result)
        barrier()
        if rank == 0:
            result["settings"] = sorted(
                [
                    setting
                    for index in range(workers)
                    for setting in read_json(shard.parent / f"rank-{index:03d}.json")[
                        "settings"
                    ]
                ],
                key=lambda r: -r["mean_cosine"],
            )
            result["gpu_workers"] = workers
            write_json(output, result)
        barrier()
    else:
        write_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--bank-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--sources",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs/sources.lock.json",
    )
    p.add_argument("--sample-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    lock = read_json(args.sources)
    config = ScoreConfig(
        model_name=lock["model"]["repo"],
        model_revision=lock["model"]["revision"],
        device=args.device,
    )
    diagnose(args.bank_dir, args.output, config, default_cache(), args.sample_size)


if __name__ == "__main__":
    main()
