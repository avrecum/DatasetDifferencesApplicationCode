"""Refactored FG-CLIP helpers from the supplied scripts, with explicit validity.

Default extraction uses the official single-image API. The optional batched
dense head mirrors the supplied detection script and is reference-gated.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import importlib.metadata
import os
from pathlib import Path

import numpy as np

from .io import digest, file_hash, read_json, write_json

SCORING_VERSION = "fgclip-masked-max-v1"


@dataclass(frozen=True)
class ScoreConfig:
    model_name: str = "qihoo360/fg-clip2-base"
    model_revision: str = "430fbc8a912c86fd4de601381b6245a0edab22f0"
    bank_revision: str = "4edc5ef4ac601a3d08225b53567b6695cbd60b5e"
    patch_budget: int = 1024
    dtype: str = "float32"
    device: str = "cuda"
    image_batch_size: int = 1
    text_chunk_size: int = 512
    reference_atol: float = 2e-4
    reference_rtol: float = 2e-4
    implementation: str = SCORING_VERSION
    preset: str = "pilot"

    def __post_init__(self):
        if min(self.patch_budget, self.image_batch_size, self.text_chunk_size) < 1:
            raise ValueError("Patch/batch/chunk sizes must be positive")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported dtype")
        if self.preset == "reproduction" and self.patch_budget != 16384:
            raise ValueError("Reproduction preset requires 16384 patches")


def load_text_embeddings(text_embeddings_dir, device="cpu", return_audit=False):
    """Same principal interface as supplied fg_clip.py; safe load and strict alignment."""
    import torch
    import torch.nn.functional as F

    root = Path(text_embeddings_dir)
    embeddings = torch.load(
        root / "embeddings.pt", map_location="cpu", weights_only=True
    )
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("Bank must be a [V,D] tensor")
    vocab = (root / "vocab.txt").read_text(encoding="utf-8").splitlines()
    # Never drop blank rows and shift vocabulary alignment.
    if any(not x.strip() for x in vocab) or len(vocab) != len(embeddings):
        raise ValueError("Blank vocabulary entry or tensor/vocabulary row mismatch")
    embeddings = embeddings.float()
    norms = embeddings.norm(dim=-1)
    if not torch.isfinite(embeddings).all() or (norms <= 0).any():
        raise ValueError("Nonfinite or zero text embeddings")
    audit = dict(
        rows=len(vocab),
        dimension=embeddings.shape[1],
        finite=True,
        duplicate_terms=len(vocab) - len(set(vocab)),
        raw_norm_min=float(norms.min()),
        raw_norm_max=float(norms.max()),
        raw_norm_mean=float(norms.mean()),
        normalized_for_scoring=True,
        embedding_sha256=file_hash(root / "embeddings.pt"),
        vocab_sha256=file_hash(root / "vocab.txt"),
        vocabulary_order_hash=digest(vocab),
        name="full supplied vocabulary (not assumed COCA-20k)",
        generation_provenance="No generation metadata supplied in pinned root repository; encoding mode initially unknown",
    )
    embeddings = F.normalize(embeddings, dim=-1).to(device)
    return (embeddings, vocab, audit) if return_audit else (embeddings, vocab)


def masked_max_scores(features, text_embeddings, valid_mask=None, text_chunk_size=512):
    """[B,P,D] x [V,D] -> [B,V], excluding padding. All-padding images -> NaN."""
    import torch
    import torch.nn.functional as F

    if (
        features.ndim != 3
        or text_embeddings.ndim != 2
        or features.shape[-1] != text_embeddings.shape[-1]
    ):
        raise ValueError("Expected [B,P,D] and [V,D] embeddings in the same space")
    if text_chunk_size < 1:
        raise ValueError("Chunk size must be positive")
    b, p, _ = features.shape
    if valid_mask is None:
        valid_mask = torch.ones((b, p), dtype=torch.bool, device=features.device)
    if valid_mask.shape != (b, p):
        raise ValueError("Patch mask shape mismatch")
    valid_mask = valid_mask.to(features.device).bool()
    valid_features = features[valid_mask]
    if (
        not torch.isfinite(valid_features).all()
        or (valid_features.float().norm(dim=-1) == 0).any()
    ):
        raise ValueError("Invalid real patch embeddings")
    if (
        not torch.isfinite(text_embeddings).all()
        or (text_embeddings.float().norm(dim=-1) == 0).any()
    ):
        raise ValueError("Invalid text embeddings")
    feats = F.normalize(
        features.float().masked_fill(~valid_mask.unsqueeze(-1), 0), dim=-1
    )
    text = F.normalize(text_embeddings.float(), dim=-1)
    out = torch.full((b, len(text)), float("nan"), device=features.device)
    if p == 0:
        return out
    for start in range(0, len(text), text_chunk_size):
        sim = feats @ text[start : start + text_chunk_size].T
        sim.masked_fill_(~valid_mask.unsqueeze(-1), -torch.inf)
        out[:, start : start + text_chunk_size] = sim.max(dim=1).values
    out[~valid_mask.any(dim=1)] = torch.nan
    return out


def compute_max_over_patches_similarity(
    patch_embeddings_list, text_embeddings, device, text_batch_size=512
):
    """Supplied helper interface preserved; failures are NaN, never zero observations."""
    import torch

    result = torch.full(
        (len(patch_embeddings_list), len(text_embeddings)), float("nan")
    )
    for i, features in enumerate(patch_embeddings_list):
        if features is not None:
            result[i] = masked_max_scores(
                features.to(device)[None],
                text_embeddings.to(device),
                text_chunk_size=text_batch_size,
            )[0].cpu()
    return result


def patch_mask(inputs, features):
    import torch

    b, p = features.shape[:2]
    mask = inputs.get("pixel_attention_mask")
    if mask is not None:
        if mask.shape[0] != b or mask.shape[1] < p:
            raise ValueError("Processor/feature mask mismatch")
        mask = mask[:, :p].bool()
    else:
        mask = torch.ones((b, p), dtype=torch.bool, device=features.device)
    if "spatial_shapes" in inputs:
        counts = inputs["spatial_shapes"].prod(dim=-1)
        if (counts > p).any():
            raise ValueError("Dense API truncated real patches")
        expected = torch.arange(p, device=features.device)[None] < counts[:, None]
        if not torch.equal(mask, expected):
            raise ValueError("Unknown processor patch ordering/padding geometry")
    return mask


def official_dense(model, inputs):
    method = getattr(model, "get_image_dense_feature", None) or getattr(
        model, "get_image_dense_features", None
    )
    if method is None:
        raise ValueError("Model lacks the official FG-CLIP dense API")
    features = method(**inputs)
    if features.ndim == 2:
        features = features[None]
    if features.ndim != 3 or features.shape[0] != inputs["pixel_values"].shape[0]:
        raise ValueError("Official dense API returned unexpected batch geometry")
    return features


def batched_dense_features(model, inputs):
    """Adapted supplied helper. Must pass official float32 reference gate before use."""
    import torch

    output = model.vision_model(
        pixel_values=inputs["pixel_values"],
        attention_mask=inputs.get("pixel_attention_mask"),
        spatial_shapes=inputs.get("spatial_shapes"),
    )
    hidden = output.last_hidden_state
    head = model.dense_feature_head
    mask = inputs.get("pixel_attention_mask")
    attention_mask = None
    if mask is not None:
        b, p = mask.shape
        attention_mask = torch.zeros(
            (b, 1, p, p), dtype=hidden.dtype, device=hidden.device
        )
        attention_mask.masked_fill_(
            ~mask.bool()[:, None, None, :], torch.finfo(hidden.dtype).min
        )
        attention_mask = attention_mask.expand(b, head.num_heads, p, p).reshape(
            -1, p, p
        )
    hidden = head.attention(hidden, hidden, hidden, attn_mask=attention_mask)[0]
    return hidden + head.mlp(head.layernorm(hidden))


class ImageInferenceDataset:
    """Portable lazy decoding/preprocessing; carries errors and stable image IDs."""

    def __init__(self, records, processor, max_num_patches=1024):
        self.records, self.processor, self.max_num_patches = (
            records,
            processor,
            max_num_patches,
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        from PIL import Image

        record = self.records[idx]
        try:
            with Image.open(record["local_path"]) as image:
                inputs = self.processor(
                    images=image.convert("RGB"),
                    max_num_patches=self.max_num_patches,
                    return_tensors="pt",
                )
            return dict(
                record_idx=idx, image_id=record["image_id"], inputs=inputs, error=None
            )
        except Exception as exc:
            return dict(
                record_idx=idx,
                image_id=record["image_id"],
                inputs=None,
                error=f"{type(exc).__name__}: {exc}",
            )


def collate_fn(batch):
    import torch

    ok = [r for r in batch if r["inputs"] is not None]
    inputs = (
        {k: torch.cat([r["inputs"][k] for r in ok]) for k in ok[0]["inputs"]}
        if ok
        else None
    )
    return dict(
        inputs=inputs, records=ok, failures=[r for r in batch if r["inputs"] is None]
    )


def _to_device(inputs, device, dtype):
    return {
        k: v.to(device=device, dtype=dtype if k == "pixel_values" else v.dtype)
        for k, v in inputs.items()
    }


def extract_dense_image_embeddings(
    image_paths, model, image_processor, device, max_num_patches=16384
):
    import torch
    import torch.nn.functional as F

    records = [{"local_path": str(p), "image_id": str(p)} for p in image_paths]
    dataset = ImageInferenceDataset(records, image_processor, max_num_patches)
    result = []
    with torch.inference_mode():
        for item in dataset:
            if item["error"]:
                result.append(None)
                continue
            inputs = _to_device(item["inputs"], device, next(model.parameters()).dtype)
            features = official_dense(model, inputs)
            result.append(
                F.normalize(
                    features[0][patch_mask(inputs, features)[0]].float(), dim=-1
                ).cpu()
            )
    return result


class ScoreCache:
    """Single-writer memory map; scores flush before an atomic validity commit.

    Invalid/uncommitted rows remain NaN logically, including after interruption.
    Fingerprint mismatch is an error, never an implicit reuse/overwrite.
    """

    def __init__(self, root, images, vocab, configuration, resume=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = configuration
        unique, rows = {}, []
        self.row_index = {}
        for im in sorted(images, key=lambda x: x["image_id"]):
            identity = im.get("pixel_hash") or im.get("content_hash") or im["image_id"]
            if identity not in unique:
                unique[identity] = len(rows)
                rows.append(im)
            self.row_index[im["image_id"]] = unique[identity]
        self.images = rows
        self.shape = (len(rows), len(vocab))
        required_bytes = int(np.prod(self.shape) * 4)
        print(
            f"Score cache: {self.shape[0]} unique images × {self.shape[1]} concepts; {required_bytes / 1e9:.3f} GB float32",
            flush=True,
        )
        self.fingerprint = digest(
            dict(
                configuration=configuration,
                vocab=vocab,
                images=[
                    {
                        k: i.get(k)
                        for k in (
                            "image_id",
                            "source_revision",
                            "content_hash",
                            "pixel_hash",
                        )
                    }
                    for i in rows
                ],
                row_index=self.row_index,
            )
        )
        meta_path = self.root / "cache.json"
        if meta_path.exists():
            meta = read_json(meta_path)
            if meta["fingerprint"] != self.fingerprint:
                raise ValueError(
                    "Incompatible score cache: model/vocabulary/preprocessing/images changed"
                )
            if not resume:
                raise FileExistsError(
                    "Score cache exists; use --resume or a new output directory"
                )
            self.state = read_json(self.root / "state.json")
            if self.shape[0] and self.shape[1]:
                self.scores = np.memmap(
                    self.root / "scores.f32",
                    mode="r+",
                    dtype="float32",
                    shape=self.shape,
                )
            else:
                self.scores = np.empty(self.shape, np.float32)
        else:
            import shutil

            if shutil.disk_usage(self.root).free < required_bytes + 10 * 1024 * 1024:
                raise OSError(
                    f"Insufficient free space for {required_bytes} score bytes"
                )
            self.state = {
                "valid": [False] * len(rows),
                "errors": {},
                "patch_geometry": {},
            }
            if self.shape[0] and self.shape[1]:
                self.scores = np.memmap(
                    self.root / "scores.f32",
                    mode="w+",
                    dtype="float32",
                    shape=self.shape,
                )
                self.scores[:] = np.nan
                self.scores.flush()
            else:
                self.scores = np.empty(self.shape, np.float32)
                (self.root / "scores.f32").touch()
            write_json(self.root / "row_index.json", self.row_index)
            write_json(self.root / "rows.json", rows)
            write_json(self.root / "vocab.json", vocab)
            self._save_state()
            write_json(
                meta_path,
                dict(
                    fingerprint=self.fingerprint,
                    configuration=configuration,
                    shape=self.shape,
                    score_bytes=int(np.prod(self.shape) * 4),
                    storage_dtype="float32",
                    single_writer=True,
                ),
            )
        self.valid = np.array(self.state["valid"], bool)
        # A commit with corrupt scores is never trusted.
        for row in np.flatnonzero(self.valid):
            if not np.isfinite(self.scores[row]).all():
                self.valid[row] = False
                self.state["valid"][row] = False
        self._save_state()

    def _save_state(self):
        write_json(self.root / "state.json", self.state)
        tmp = self.root / "valid.tmp.npy"
        np.save(tmp, np.asarray(self.state["valid"], bool))
        os.replace(tmp, self.root / "valid.npy")

    def write(self, row, scores, geometry=None):
        scores = np.asarray(scores, np.float32)
        if scores.shape != (self.shape[1],) or not np.isfinite(scores).all():
            raise ValueError("Cannot commit invalid score row")
        self.scores[row] = scores
        if hasattr(self.scores, "flush"):
            self.scores.flush()
        self.valid[row] = self.state["valid"][row] = True
        self.state["errors"].pop(str(row), None)
        if geometry is not None:
            self.state["patch_geometry"][str(row)] = geometry
        self._save_state()

    def fail(self, row, error):
        self.valid[row] = self.state["valid"][row] = False
        self.state["errors"][str(row)] = str(error)
        self._save_state()


def open_scores(root):
    root = Path(root)
    meta = read_json(root / "cache.json")
    shape = tuple(meta["shape"])
    scores = (
        np.memmap(root / "scores.f32", mode="r", dtype="float32", shape=shape)
        if all(shape)
        else np.empty(shape)
    )
    state = read_json(root / "state.json")
    valid = np.array(state["valid"], bool)
    for r in np.flatnonzero(valid):
        if not np.isfinite(scores[r]).all():
            raise ValueError("Committed score row is corrupt")
    return (
        scores,
        read_json(root / "row_index.json"),
        valid,
        read_json(root / "vocab.json"),
        meta,
        state,
    )


def load_model(config, cache):
    import torch
    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

    if (
        config.device.startswith("cuda")
        and not os.environ.get("SLURM_JOB_ID")
        and os.environ.get("SLURM_CLUSTER_NAME")
    ):
        raise RuntimeError("GPU scoring on this cluster requires a Slurm allocation")
    kwargs = dict(
        revision=config.model_revision,
        code_revision=config.model_revision,
        cache_dir=str(Path(cache) / "hub"),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
            **kwargs,
        )
        .to(config.device)
        .eval()
    )
    model.requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(config.model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, **kwargs)
    return model, processor, tokenizer


def audit_text_modes(model, tokenizer, text, vocab, device):
    import torch
    import torch.nn.functional as F

    indices = np.linspace(0, len(vocab) - 1, min(12, len(vocab)), dtype=int)
    rows = []
    with torch.inference_mode():
        for mode, length in (("box", 64), ("short", 64), ("long", 196)):
            words = [vocab[i].lower() for i in indices]
            inputs = tokenizer(
                words,
                padding="max_length",
                max_length=length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            feats = F.normalize(
                model.get_text_features(**inputs, walk_type=mode).float(), dim=-1
            )
            cos = (feats * text[indices].to(device)).sum(dim=-1).cpu().numpy()
            rows.append(
                dict(
                    walk_type=mode,
                    max_length=length,
                    lowercased=True,
                    words=words,
                    cosine_to_bank=cos.tolist(),
                    mean_cosine=float(cos.mean()),
                    min_cosine=float(cos.min()),
                )
            )
    return dict(
        settings=rows,
        conclusion="Diagnostic only: high agreement supports compatibility, but cannot recover missing generation provenance. Original bank retained.",
    )


def validate_reference(model, processor, records, text, config):
    import torch

    dtype = getattr(torch, config.dtype)
    records = records[: max(2, min(config.image_batch_size, 4))]
    dataset = ImageInferenceDataset(records, processor, config.patch_budget)
    items = [dataset[i] for i in range(len(dataset))]
    if any(x["error"] for x in items):
        raise RuntimeError("Cannot validate scoring reference on failed images")
    indices = np.linspace(0, len(text) - 1, min(128, len(text)), dtype=int)
    bank = text[indices].to(config.device)
    references = []
    with torch.inference_mode():
        model.float()
        for item in items:
            inputs = _to_device(item["inputs"], config.device, torch.float32)
            features = official_dense(model, inputs)
            references.append(
                masked_max_scores(
                    features, bank, patch_mask(inputs, features), config.text_chunk_size
                )[0].cpu()
            )
        model.to(dtype=dtype)
        outputs = []
        if config.image_batch_size > 1:
            inputs = _to_device(collate_fn(items)["inputs"], config.device, dtype)
            features = batched_dense_features(model, inputs)
            outputs = masked_max_scores(
                features, bank, patch_mask(inputs, features), config.text_chunk_size
            ).cpu()
        else:
            for item in items:
                inputs = _to_device(item["inputs"], config.device, dtype)
                features = official_dense(model, inputs)
                outputs.append(
                    masked_max_scores(
                        features,
                        bank,
                        patch_mask(inputs, features),
                        config.text_chunk_size,
                    )[0].cpu()
                )
            outputs = torch.stack(outputs)
        reference = torch.stack(references)
        error = float((outputs - reference).abs().max())
        passed = torch.allclose(
            outputs, reference, atol=config.reference_atol, rtol=config.reference_rtol
        )
    audit = dict(
        passed=passed,
        max_absolute_score_error=error,
        atol=config.reference_atol,
        rtol=config.reference_rtol,
        images=[r["image_id"] for r in records],
        concepts=len(indices),
        reference="official dense API, float32, one image",
        candidate="supplied detection dense-head batch"
        if config.image_batch_size > 1
        else "official dense API",
        dtype=config.dtype,
        patch_budget=config.patch_budget,
    )
    if not passed:
        raise RuntimeError(f"Optimized/reference equivalence failed: {audit}")
    return audit


def score_manifest(manifest, bank_dir, output, config, cache_root, resume=False):
    import torch

    text, vocab, bank_audit = load_text_embeddings(bank_dir, "cpu", True)
    versions = {
        k: importlib.metadata.version(k)
        for k in ("torch", "transformers", "Pillow", "numpy")
    }
    configuration = dict(**asdict(config), bank=bank_audit, library_versions=versions)
    required = {
        c
        for e in manifest["events"]
        if e["sampled"] and not e["exclusion_reason"]
        for c in e["candidate_ids"]
    }
    images = [i for i in manifest["images"] if i["image_id"] in required]
    for im in images:
        if (
            im["status"] == "valid"
            and file_hash(im["local_path"]) != im["content_hash"]
        ):
            raise ValueError(
                "Image bytes changed after preparation; prepare again before scoring"
            )
    cache = ScoreCache(output, images, vocab, configuration, resume)
    write_json(Path(output) / "bank_audit.json", bank_audit)
    pending = [
        i
        for i in range(len(cache.images))
        if not cache.valid[i] and cache.images[i]["status"] == "valid"
    ]
    for i, im in enumerate(cache.images):
        if im["status"] != "valid":
            cache.fail(i, im.get("error") or "unavailable image")
    if not pending:
        return cache
    if config.device.startswith("cuda"):
        # Explicitly block known login hosts even without cluster environment variables.
        import socket

        if not os.environ.get("SLURM_JOB_ID") and socket.gethostname().startswith(
            ("jpbl", "login")
        ):
            raise RuntimeError(
                "Do not run GPU inference on a login node; use scripts/pilot.sbatch"
            )
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model, processor, tokenizer = load_model(config, cache_root)
    write_json(
        Path(output) / "text_mode_audit.json",
        audit_text_modes(model, tokenizer, text, vocab, config.device),
    )
    records = [cache.images[i] for i in pending]
    reference = validate_reference(model, processor, records, text, config)
    write_json(Path(output) / "reference_validation.json", reference)
    dataset = ImageInferenceDataset(records, processor, config.patch_budget)
    text = text.to(config.device)
    dtype = getattr(torch, config.dtype)
    with torch.inference_mode():
        for start in range(0, len(records), config.image_batch_size):
            batch = collate_fn(
                [
                    dataset[i]
                    for i in range(
                        start, min(start + config.image_batch_size, len(records))
                    )
                ]
            )
            for bad in batch["failures"]:
                cache.fail(pending[bad["record_idx"]], bad["error"])
            if batch["inputs"] is None:
                continue
            try:
                inputs = _to_device(batch["inputs"], config.device, dtype)
                features = (
                    batched_dense_features
                    if config.image_batch_size > 1
                    else official_dense
                )(model, inputs)
                mask = patch_mask(inputs, features)
                scores = (
                    masked_max_scores(features, text, mask, config.text_chunk_size)
                    .cpu()
                    .numpy()
                )
                for b, item in enumerate(batch["records"]):
                    shape = (
                        inputs["spatial_shapes"][b].tolist()
                        if "spatial_shapes" in inputs
                        else None
                    )
                    geometry = dict(
                        valid_patches=int(mask[b].sum()),
                        spatial_shape=shape,
                        patch_budget=config.patch_budget,
                        processor=type(processor).__name__,
                    )
                    cache.write(pending[item["record_idx"]], scores[b], geometry)
            except torch.OutOfMemoryError:
                raise  # Never silently change patch budgets or mix resolutions.
            except Exception as exc:
                for item in batch["records"]:
                    cache.fail(
                        pending[item["record_idx"]],
                        f"inference: {type(exc).__name__}: {exc}",
                    )
            print(
                f"Scored {min(start + config.image_batch_size, len(records))}/{len(records)} unique pending images",
                flush=True,
            )
    return cache
