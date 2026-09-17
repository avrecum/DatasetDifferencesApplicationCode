"""Deterministic offline end-to-end fixture. NEVER presented as real model evidence."""

from pathlib import Path
import numpy as np
from PIL import Image

from .adapters import imagereward, pickapic
from .io import save_manifest
from .sources import validate_image
from .splits import assign_splits, sample_components
from .scoring import ScoreCache


def build_fixture(output, dataset="imagereward", null=False):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    image_rows = []
    events = []
    rng = np.random.default_rng(42)
    for g in range(20):
        split = "train" if g < 14 else "validation" if g < 17 else "test"
        for k in range(4):
            uid = f"g{g}_{k}"
            path = (root / f"{uid}.webp").resolve()
            # Distinct deterministic pixels ensure synthetic identities do not collapse.
            pixels = rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path, lossless=True)
            common = dict(prompt=f"synthetic prompt {g}", local_path=str(path))
            rows.append(
                dict(
                    **common,
                    prompt_id=str(g),
                    image_path=uid + ".webp",
                    rank=k + 1,
                    image_amount_in_total=4,
                    original_split=split,
                )
            )
            image_rows.append(
                dict(
                    **common,
                    image_uid=uid,
                    negative_prompt=None,
                    url=None,
                    model_id="synthetic-generator",
                )
            )
        events.append(
            dict(
                ranking_id=g,
                user_id=g % 3,
                prompt=f"synthetic prompt {g}",
                best_image_uid=f"g{g}_0",
                **{f"image_{k + 1}_uid": f"g{g}_{k}" for k in range(4)},
            )
        )
    m = imagereward(rows) if dataset == "imagereward" else pickapic(events, image_rows)
    m["provenance"]["synthetic"] = True
    for im in m["images"]:
        im.update(validate_image(im["local_path"]), status="valid")
    assign_splits(m)
    sample_components(m, 0)
    save_manifest(root / "manifest", m)
    config = dict(
        model_name="SYNTHETIC_NO_ENCODER",
        model_revision="synthetic",
        patch_budget=1,
        implementation="fixture-v1",
    )
    vocab = ["known_positive", "known_negative", "constant_null", "irrelevant_balanced"]
    cache = ScoreCache(root / "scores", m["images"], vocab, config)
    for i, im in enumerate(cache.images):
        uid = im["image_id"].split(":", 1)[1].replace(".webp", "")
        g, k = map(int, uid[1:].split("_"))
        value = 1 - k / 3
        x = (
            np.array([value, 1 - value, 0.5, (g % 2) * 0.1])
            if not null
            else np.ones(4) * 0.5
        )
        cache.write(i, x, dict(valid_patches=1, spatial_shape=[1, 1], patch_budget=1))
    return m
