"""Similarity maps on the actual processed patch geometry, never segmentation masks."""

from dataclasses import fields
from pathlib import Path

import numpy as np

from .io import digest, read_json, write_json
from .scoring import (
    ScoreConfig,
    load_model,
    load_text_embeddings,
    ImageInferenceDataset,
    official_dense,
    patch_mask,
    _to_device,
    open_scores,
)


def heatmaps(manifest, score_dir, analysis_dir, bank_dir, cache_root, concept_limit=3):
    import torch
    import torch.nn.functional as F
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .parallel import context, barrier

    rank, workers = context()

    root = Path(analysis_dir) / "heatmaps"
    root.mkdir(exist_ok=True)
    examples = read_json(Path(analysis_dir) / "gallery_selection.json")["examples"]
    chosen = []
    seen = set()
    for row in examples:
        if row["concept_id"] not in seen and row["selection"] == "random_audit":
            chosen.append(row)
            seen.add(row["concept_id"])
            if len(chosen) >= concept_limit:
                break
    if not chosen:
        write_json(
            root / "index.json",
            dict(status="unavailable: no eligible gallery pairs", maps=[]),
        )
        return
    if workers > 1 and len(chosen) < workers:
        raise ValueError("Choose at least one heatmap concept per GPU worker")
    chosen = chosen[rank::workers]
    scores, index, valid, vocab, meta, state = open_scores(score_dir)
    if meta["configuration"].get("model_name") == "SYNTHETIC_NO_ENCODER":
        write_json(
            root / "index.json",
            dict(status="not generated for synthetic scores", maps=[]),
        )
        return
    keys = {f.name for f in fields(ScoreConfig)}
    cfg = ScoreConfig(**{k: v for k, v in meta["configuration"].items() if k in keys})
    model, processor, _ = load_model(cfg, cache_root)
    model.to(dtype=getattr(torch, cfg.dtype))
    if type(processor).__name__ not in (
        "Siglip2ImageProcessorFast",
        "Siglip2ImageProcessor",
    ):
        raise ValueError("Unknown processor geometry: cannot produce faithful overlay")
    bank, words = load_text_embeddings(bank_dir, cfg.device)
    images = {i["image_id"]: i for i in manifest["images"]}
    artifacts = []
    with torch.inference_mode():
        for row in chosen:
            maps = []
            views = []
            geometries = []
            for side in ("preferred", "rejected"):
                iid = row[side + "_image_id"]
                item = ImageInferenceDataset(
                    [images[iid]], processor, cfg.patch_budget
                )[0]
                if item["error"]:
                    raise ValueError(item["error"])
                inputs = _to_device(
                    item["inputs"], cfg.device, getattr(torch, cfg.dtype)
                )
                features = official_dense(model, inputs)
                mask = patch_mask(inputs, features)[0]
                h, w = map(int, inputs["spatial_shapes"][0].tolist())
                similarity = (
                    (
                        F.normalize(features[0][mask].float(), dim=-1)
                        @ bank[row["concept_id"]]
                    )
                    .cpu()
                    .numpy()
                    .reshape(h, w)
                )
                # Invert the documented SigLIP2 row-major HWC patch packing and
                # normalization. This shows the exact resized model input, not an
                # assumed crop/resize of the source image.
                patch = int(processor.patch_size)
                pixels = item["inputs"]["pixel_values"][0, : h * w].float().numpy()
                pixels = (
                    pixels.reshape(h, w, patch, patch, 3)
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(h * patch, w * patch, 3)
                )
                pixels = pixels * np.array(processor.image_std) + np.array(
                    processor.image_mean
                )
                pixels = np.clip(pixels, 0, 1)
                expected = float(scores[index[iid], row["concept_id"]])
                if not np.isclose(
                    similarity.max(),
                    expected,
                    atol=cfg.reference_atol,
                    rtol=cfg.reference_rtol,
                ):
                    raise ValueError(
                        "Heatmap maximum disagrees with cached image score"
                    )
                maps.append(similarity)
                views.append(pixels)
                geometries.append(
                    dict(
                        image_id=iid,
                        spatial_shape=[h, w],
                        patch_size=patch,
                        valid_patches=h * w,
                        maximum=float(similarity.max()),
                        argmax_patch=list(
                            map(
                                int,
                                np.unravel_index(similarity.argmax(), similarity.shape),
                            )
                        ),
                    )
                )
            low = min(float(m.min()) for m in maps)
            high = max(float(m.max()) for m in maps)
            fig, axes = plt.subplots(2, 2, figsize=(9, 8))
            for k, side in enumerate(("preferred", "rejected")):
                axes[0, k].imshow(views[k])
                axes[0, k].set_title(f"{side}: actual processed input")
                axes[1, k].imshow(views[k])
                shown = axes[1, k].imshow(
                    maps[k],
                    extent=[0, views[k].shape[1], views[k].shape[0], 0],
                    cmap="viridis",
                    alpha=0.65,
                    vmin=low,
                    vmax=high,
                    interpolation="nearest",
                )
                y, x = geometries[k]["argmax_patch"]
                from matplotlib.patches import Rectangle

                patch = geometries[k]["patch_size"]
                axes[1, k].add_patch(
                    Rectangle(
                        (x * patch, y * patch),
                        patch,
                        patch,
                        fill=False,
                        edgecolor="red",
                        lw=2,
                    )
                )
                for ax in axes[:, k]:
                    ax.axis("off")
            fig.suptitle(
                f"{row['concept']} — unverified model similarity; not segmentation ground truth"
            )
            fig.colorbar(
                shown,
                ax=axes[1, :].tolist(),
                label="normalized text–patch dot product",
                shrink=0.8,
            )
            name = digest([row["concept_id"], row["event_id"]])[:20]
            fig.savefig(root / (name + ".png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            for k, side in enumerate(("preferred", "rejected")):
                np.save(root / (name + f"_{side}.npy"), maps[k])
            artifacts.append(
                dict(
                    concept=row["concept"],
                    event_id=row["event_id"],
                    selection="first seeded random-audit pair",
                    figure=name + ".png",
                    geometry=geometries,
                    scale=[low, high],
                )
            )
    result = dict(
        status="unverified model similarity maps",
        geometry="inverse processor HWC patch packing; valid prefix patches only; exact processed view",
        maps=artifacts,
        worker_rank=rank,
    )
    if workers > 1:
        write_json(root / f"index-rank-{rank:03d}.json", result)
        barrier()
        if rank == 0:
            result["maps"] = [
                m
                for i in range(workers)
                for m in read_json(root / f"index-rank-{i:03d}.json")["maps"]
            ]
            result["gpu_workers"] = workers
            write_json(root / "index.json", result)
        barrier()
    else:
        write_json(root / "index.json", result)
