from dataclasses import asdict
import pytest

from preference_diff.scoring import (
    ScoreCache,
    ScoreConfig,
    masked_max_scores,
    compute_max_over_patches_similarity,
    load_text_embeddings,
    open_scores,
)
from preference_diff.sources import safe_member, extract_required


def test_masked_padding_chunk_reference_and_failure():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(7)
    f = torch.randn(3, 7, 8, generator=g)
    t = torch.randn(11, 8, generator=g)
    mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0]],
        dtype=torch.bool,
    )
    x = masked_max_scores(f, t, mask, 3)
    ref = (
        torch.nn.functional.normalize(f[:2], dim=-1)
        @ torch.nn.functional.normalize(t, dim=-1).T
    )
    ref.masked_fill_(~mask[:2, :, None], -torch.inf)
    torch.testing.assert_close(x[:2], ref.max(1).values)
    assert torch.isnan(x[2]).all()
    torch.testing.assert_close(x, masked_max_scores(f, t, mask, 11), equal_nan=True)
    failed = compute_max_over_patches_similarity([None, f[0]], t, "cpu", 3)
    assert torch.isnan(failed[0]).all() and torch.isfinite(failed[1]).all()


def test_padded_high_similarity_cannot_win():
    torch = pytest.importorskip("torch")
    x = masked_max_scores(
        torch.tensor([[[-1.0, 0.0], [1.0, 0.0]]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1, 0]]),
    )
    assert x.item() == -1


def test_safe_bank_validation_and_alignment(tmp_path):
    torch = pytest.importorskip("torch")
    (tmp_path / "vocab.txt").write_text("a\nb\n")
    torch.save(torch.tensor([[1.0, 0.0], [0.0, 2.0]]), tmp_path / "embeddings.pt")
    t, v, a = load_text_embeddings(tmp_path, return_audit=True)
    assert v == ["a", "b"] and a["raw_norm_max"] == 2
    torch.testing.assert_close(t.norm(dim=1), torch.ones(2))
    (tmp_path / "vocab.txt").write_text("a\n\nb\n")
    with pytest.raises(ValueError):
        load_text_embeddings(tmp_path)
    (tmp_path / "vocab.txt").write_text("a\nb\n")
    torch.save(torch.zeros(2, 2), tmp_path / "embeddings.pt")
    with pytest.raises(ValueError):
        load_text_embeddings(tmp_path)


@pytest.mark.parametrize(
    "changed", ["vocabulary", "model", "patches", "dtype", "content"]
)
def test_cache_fingerprint_resume_and_atomic_validity(tmp_path, changed):
    ims = [
        dict(
            image_id="a",
            content_hash="hash",
            pixel_hash="pixels",
            source_revision="rev",
        ),
        dict(
            image_id="alias",
            content_hash="other",
            pixel_hash="pixels",
            source_revision="rev",
        ),
        dict(image_id="b", content_hash="b", pixel_hash="b", source_revision="rev"),
    ]
    cfg = asdict(ScoreConfig(device="cpu"))
    v = ["a", "b"]
    c = ScoreCache(tmp_path, ims, v, cfg)
    c.write(0, [0.1, 0.2])
    c.fail(1, "decode failed")
    assert c.row_index["a"] == c.row_index["alias"]
    c.scores[1] = [0.9, 0.9]
    c.scores.flush()  # simulate interruption before validity commit
    resumed = ScoreCache(tmp_path, ims, v, cfg, True)
    assert resumed.valid.tolist() == [True, False]
    assert open_scores(tmp_path)[2].tolist() == [True, False]
    if changed == "vocabulary":
        v = v[::-1]
    elif changed == "model":
        cfg["model_revision"] = "new"
    elif changed == "patches":
        cfg["patch_budget"] = 16384
    elif changed == "dtype":
        cfg["dtype"] = "bfloat16"
    elif changed == "content":
        ims[0]["content_hash"] = "changed"
    with pytest.raises(ValueError):
        ScoreCache(tmp_path, ims, v, cfg, True)


def test_archive_paths_and_requested_members(tmp_path):
    import zipfile
    from PIL import Image

    for bad in ("../x", "/absolute", "x/../a", "a\\b", "C:x"):
        with pytest.raises(ValueError):
            safe_member(bad)
    im = tmp_path / "x.webp"
    Image.new("RGB", (2, 3)).save(im)
    z = tmp_path / "safe.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.write(im, "folder/x.webp")
    out = extract_required(z, ["images/train/train_1/x.webp"], tmp_path / "out")
    assert len(out) == 1


@pytest.mark.gpu
def test_real_gpu_reference():
    import os

    if not os.environ.get("PREFERENCE_GPU_TEST_MANIFEST"):
        pytest.skip(
            "Set PREFERENCE_GPU_TEST_MANIFEST inside a GPU allocation for actual FG-CLIP equivalence"
        )
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from preference_diff.io import load_manifest
    from preference_diff.scoring import load_model, validate_reference
    from preference_diff.sources import default_cache

    m = load_manifest(os.environ["PREFERENCE_GPU_TEST_MANIFEST"])
    cfg = ScoreConfig(image_batch_size=2)
    model, processor, _ = load_model(cfg, default_cache())
    text, _ = load_text_embeddings(os.environ["PREFERENCE_GPU_TEST_BANK"])
    available = sorted(
        [i for i in m["images"] if i["status"] == "valid"],
        key=lambda i: i.get("aspect_ratio", 1),
    )
    records = [available[0], available[-1]]
    result = validate_reference(model, processor, records, text, cfg)
    if os.environ.get("PREFERENCE_GPU_TEST_REPORT"):
        from preference_diff.io import write_json

        write_json(os.environ["PREFERENCE_GPU_TEST_REPORT"], result)
    assert result["passed"]
