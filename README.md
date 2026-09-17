# Human-preference dataset diffing

Frozen FG-CLIP2 patch embeddings discover visual concepts associated with recorded human preference **within** ImageRewardDB and the linked four-candidate Pick-a-Pic release. This implements the missing human-preference application for *Detecting Fine-Grained Differences between Image Datasets*. It does not train a reward model, generate images, or substitute model predictions for human labels.

See the [latest 500-prompt results and handoff](reports/CONTINUATION.md), [discrimination plot](reports/pilot500/imagereward/discrimination_summary.png), [paired gallery](reports/pilot500/imagereward/gallery.html), [source/method review](docs/METHOD_AND_SOURCE_REVIEW.md), [implementation plan](IMPLEMENTATION_PLAN.md), and [pinned revisions](configs/sources.lock.json). The [earlier 100-prompt handoff](reports/HANDOFF.md) remains available. Large data, score matrices and environments are ignored by Git. Actual run outputs live under `runs/` or a user-selected output path. The committed summaries identify what ran and what remains unavailable.

## Method and experiment design

For every supplied concept, normalize its text vector and each real dense patch vector, then take the maximum text–patch dot product. Padded patches never participate. Positive differences indicate higher similarity in preferred images; they are **not calibrated concept-presence probabilities**.

The primary effect is the mean preferred-minus-rejected score, averaged equally over comparisons within each event, events within each normalized prompt, then prompts. Correspondingly weighted preferred/rejected means have exactly that difference. Pair retention supports correct weighting and dependence handling; pairing alone does not change this algebraic point estimate. Reported paired concordance counts preferred-higher as 1, exact ties as 0.5, and preferred-lower as 0. A configurable numerical tie tolerance is frozen per analysis (default 0).

- **ImageReward:** lower rank is preferred. Use all strict ranked pairs and retain rank ties without inventing comparisons. Complete groups only, including all annotated candidates. Reuse the same image scores for best-versus-rest and best-versus-worst sensitivities. Overall/alignment/fidelity ratings remain metadata, not new labels.
- **Pick-a-Pic:** join the four displayed UIDs to the image table. A valid selected UID produces three winner-versus-rest comparisons. Exclude `none`, invalid selections, unresolved candidates and prompt mismatches. Non-selected images are not ordered. An image can win and lose in different events. Equal event weighting prevents repeated/candidate-rich events from dominating; exact duplicate event copies are removed by event ID while independent judgments remain.
- **Matching:** Unicode NFC and whitespace collapse, retaining case and punctuation. All Pick-a-Pic candidates must match the event prompt and have compatible negative prompts. All-null negative prompts stay explicitly unknown. `--cohort event` is a separately labeled broader sensitivity. No automatic fallback or partial-event analysis.
- **Independence:** connected components join normalized prompts, image UIDs and available byte/decoded-pixel hashes. ImageReward uses official train/validation/test as discovery/validation/test, quarantining cross-boundary components. Pick-a-Pic uses fixed-seed PCG64 shuffling of sorted independent components with largest-remainder 70/15/15 allocation. Content hashes are checked for downloaded images; metadata identities/prompts are checked over all metadata.
- **Discovery/evaluation:** strictly positive top ten and strictly negative top ten on discovery only. `frozen_candidates.json` is written before test aggregation. Directions and main-table order remain frozen even when features fail validation. No threshold optimization. The joint concept score is an equal mean of discovery-signed features, with held-out paired evaluation and no fitted weights.
- **Uncertainty:** 2,000 component-bootstrap resamples for selected features, recomputing the original prompt-weighted estimand. These are per-concept intervals, not simultaneous guarantees. Fewer than two clusters yields unavailable intervals. A conservative annotator-connected-component sensitivity is generated where user IDs exist; it may leave too few clusters.
- **Interpretability:** paired galleries include distinct-prompt supportive extremes, counterexamples and seeded random audit examples, plus blank annotation CSVs. Heatmaps show exact processed input geometry and raw model similarity, not segmentation truth. All visual evidence is labeled human-unverified.
- **Transfer:** join the full vocabulary, plot signed effects, report signed rank correlation/top-k overlap, and evaluate each dataset's frozen candidates on the other's test data after excluding components overlapping donor discovery prompts/hashes. ImageReward best-versus-rest has a separate comparison. A significant effect in only one dataset is not a between-dataset significance test.

Confounder outputs cover available generator identity, same-model versus mixed-model events, ImageReward categories, resolution/aspect ratio, generation settings, patch counts and equal-geometry event sensitivities. No unavailable model or annotator identities are inferred. Prompt matching is not causal identification.

Optional `analyze --diversity-correlation 0.95` writes a separate presentation list using discovery-image activation correlations, with every suppressed term mapped to its retained term. The unmodified frozen ranking remains the primary analysis. This filter is not Residual-OMP.

## Environment

Python 3.10+ (tested on 3.12). For a fresh isolated environment with a platform-appropriate Torch installation:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[scoring,test]'
python -m pytest -q
```

The pinned model's remote Python code is reviewed in the source notes. Scoring pins Transformers 4.57.6, tokenizers 0.22.2 and huggingface-hub 0.36.2; it uses `trust_remote_code` with an immutable model/code revision and safetensors weights. No dataset Python loader is executed. The existing cluster installation uses a local `.venv` with its own model-library versions and a `.pth` reference to the pre-existing Torch/numerical runtime, without modifying that runtime. `scripts/setup_cluster_env.sh` reproduces this optional cluster setup.

Scoring must run inside a Slurm allocation on HPC. Do not run the GPU commands on a login node. Metadata inspection, image downloading, tests and analysis can run on CPU. Prefetch all model/tokenizer files on a network-enabled node: scoring loads only local pinned files, which is required on this cluster's network-isolated compute nodes.

The scheduler scripts request and use **all four GPUs per node**. `torchrun` starts one process per GPU, assigns identical-content image groups to one worker, writes separate worker caches and merges committed rows on rank zero. The result keeps the same numerical scorer and statistical weights. Each worker checks its scores against the official FP32 API. Heatmaps are also distributed, with four selected concepts by default in the scheduler scripts. GPU utilization is sampled every two seconds; worker image counts and reference checks are saved. CPU-only aggregation and downloads do not reserve a separate GPU node.

Fresh cluster jobs use image batches of 16 and concept chunks of 2,048, with two bounded CPU prefetch batches and atomic batch score commits. Override with `PREFERENCE_IMAGE_BATCH` and `PREFERENCE_TEXT_CHUNK`. To resume or reproduce the saved primary pilots, explicitly set `PREFERENCE_IMAGE_BATCH=1 PREFERENCE_TEXT_CHUNK=512`; their cache fingerprints retain those original settings. CLI defaults remain the single-image reference configuration. Batch verification uses separate score directories and never silently imports incompatible scores.

```bash
export HF_HOME="$PROJECT/avrecum/.hf"
.venv/bin/python -m preference_diff inspect --output reports/source_inspection
.venv/bin/python -m preference_diff prefetch --output reports/source_inspection
```

The complete bank has **54,030 × 768** entries, not an assumed COCA-20k truncation. `bank_audit.json` records hashes, dimensions, norms and alignment. `text_mode_audit.json` compares twelve evenly spaced entries re-encoded with documented box/short/long settings; this diagnoses compatibility without replacing the original bank. Check that artifact before interpreting concepts.

The [expanded version-2 template diagnostic](reports/source_inspection/text_bank_template_diagnostic.json) compares 32 entries against 66 fixed template/mode/mask settings. Templated box text without a mask agrees much more closely than bare words (mean cosine 0.9656). Exact generation provenance is still unconfirmed; the bank remains unchanged. Pass `report --bank-diagnostic reports/source_inspection/text_bank_template_diagnostic.json` to include the compatible diagnostic in generated reports.

## Reproducible runs

The offline fixture generates synthetic score data and small synthetic images purely for engineering checks. It requires neither network nor GPU and labels every report as synthetic:

```bash
.venv/bin/python -m preference_diff fixture --output runs/offline_fixture/imagereward
.venv/bin/python -m preference_diff fixture --dataset pickapic --output runs/offline_fixture/pickapic
.venv/bin/python -m preference_diff fixture --null-fixture --output runs/offline_null
```

Prepare metadata before images (`--metadata-only` is available). Set a smoke limit of 10, a pilot limit up to 500, or 0 for all eligible prompt groups. Sampling preserves complete dependency components and never replenishes image failures. Pilot limits use 70/15/15 prompt caps inside the fixed splits; these differ from ImageReward's original split proportions.

```bash
export HF_HOME="$PROJECT/avrecum/.hf"
PREFERENCE_RUN_ROOT="$SCRATCH/avrecum/preference_diff/pilot500"
# If scratch quota is unavailable, choose a project-backed directory such as runs/pilot500.
for dataset in imagereward pickapic; do
  .venv/bin/python -m preference_diff prepare --dataset "$dataset" \
    --output "$PREFERENCE_RUN_ROOT/$dataset" --group-limit 500 --seed 42
done
mkdir -p reports/logs
sbatch scripts/pilot.sbatch "$PREFERENCE_RUN_ROOT"
```

On this deployment, the original Pick-a-Pic AWS URLs return 403. The run reports unavailable effects instead of changing releases. To supply accessible copies of the **same original UIDs**, store exactly one `UID.png`, `.jpg`, `.jpeg` or `.webp` file per candidate and prepare using:

```bash
.venv/bin/python -m preference_diff prepare --dataset pickapic \
  --output "$PREFERENCE_RUN_ROOT/pickapic" --group-limit 500 --resume \
  --local-image-root /path/to/original_uid_images
```

ImageReward local roots must mirror `images/train/train_N/...webp` metadata paths. Local metadata can be supplied with `--metadata`, plus `--image-metadata` for Pick-a-Pic. Preserve `original_split` on ImageReward rows. Exact source revisions default to the lock file; pass another reviewed lock with `--sources` when needed.

If restored images change an existing failed run's image hashes, its score cache is deliberately incompatible. Use a fresh run root (or a fresh `--score-dir` and `--analysis-dir`) for scoring and analysis; `--resume` does not bypass content-fingerprint checks. The full-run commands below already use a fresh root.

Inside a GPU allocation, the individual steps and the end-to-end command are:

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 --module preference_diff score --dataset imagereward \
  --output "$PREFERENCE_RUN_ROOT/imagereward" --device cuda --dtype float32 \
  --image-batch-size 1 --text-chunk-size 512 --patch-budget 1024 --resume
.venv/bin/python -m preference_diff analyze --output "$PREFERENCE_RUN_ROOT/imagereward" \
  --bootstrap 2000 --top-k 10 --seed 42
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 --module preference_diff report --output "$PREFERENCE_RUN_ROOT/imagereward" --heatmaps 4
.venv/bin/python -m preference_diff compare \
  --left "$PREFERENCE_RUN_ROOT/imagereward" --right "$PREFERENCE_RUN_ROOT/pickapic" \
  --output "$PREFERENCE_RUN_ROOT/comparison"
# For this four-GPU cluster, scripts/pilot.sbatch orchestrates the end-to-end run.
# The `run` subcommand remains available for a single-device non-cluster environment.
```

`--preset pilot` uses 1,024 patches. `--preset reproduction` uses the supplied script's explicit 16,384-patch budget, at substantially greater memory cost. OOM is fatal; the pipeline never silently changes resolution. `--image-batch-size >1` enables the adapted dense-head batch path only after a float32 official single-image reference check. Dtypes and reference tolerances are explicit and saved. FP32 defaults use `atol=rtol=2e-4`; lower precision may require a separately justified tolerance. No failed equivalence check is accepted automatically.

### Full experiment

Once original Pick-a-Pic image access is restored, the following prepares the full cohorts and runs the same pipeline. Choose storage with sufficient quota and scheduler time appropriate to the full population:

```bash
export HF_HOME="$PROJECT/avrecum/.hf"
PREFERENCE_FULL="$SCRATCH/avrecum/preference_diff/full"
.venv/bin/python -m preference_diff prefetch --output reports/source_inspection
.venv/bin/python -m preference_diff prepare --dataset imagereward \
  --output "$PREFERENCE_FULL/imagereward" --group-limit 0 --seed 42
.venv/bin/python -m preference_diff prepare --dataset pickapic \
  --output "$PREFERENCE_FULL/pickapic" --group-limit 0 --seed 42 \
  --local-image-root /path/to/original_uid_images
mkdir -p reports/logs
sbatch --time=12:00:00 scripts/pilot.sbatch "$PREFERENCE_FULL"
```

The pilot script defaults to the same 1,024-patch FP32 configuration for both datasets. A separate higher-resolution run must use distinct score/output directories, with matching settings on both datasets. Resume is explicit: manifests reject changed preparation settings; score caches reject vocabulary/order, embedding/image hashes, revisions, preprocessing, dtype, library or implementation changes. A resumed run retries invalid/uncommitted rows. Do not run two writers on the same score directory.

To enlarge a pilot without recomputing matching images, add `score --reuse-scores runs/pilot100/imagereward/scores`, or pass that path as the second argument to `scripts/imagereward_pilot.sbatch` with `PREFERENCE_IMAGE_BATCH=1 PREFERENCE_TEXT_CHUNK=512`. Reuse requires identical complete scoring configuration and vocabulary plus matching decoded-content or byte hashes. New identities alone do not authorize reuse. Imported rows and source fingerprints are recorded. Four worker directories are independent single-writer caches; their final merge is coordinated automatically. Keeping worker shards alongside the merged matrix uses approximately twice the base score storage.

The bounded text-only bank diagnostic can be reproduced with `sbatch scripts/bank_diagnostics.sbatch`. It tests fixed template/mode settings without consulting preference labels or replacing the bank. The diagnostic script also runs a separately prepared four-GPU equivalence fixture if requested; see its arguments and the continuation report for the actual run.

## Artifacts and storage

Each run has:

| Directory | Contents |
|---|---|
| `manifest/` | `images.jsonl`, `events.jsonl`, `comparisons.jsonl`, complete metadata audit and provenance |
| `scores/` | FP32 memory map, stable UID→row map, explicit validity mask/state, fingerprint/configuration, bank/text-mode/reference audits, patch counts |
| `analysis/` | all signed concepts CSV, frozen candidate JSON, held-out top concepts/intervals/counts, joint score, exact valid comparisons, confounder audits |
| `analysis/sensitivity/` | ImageReward best-versus-rest and best-versus-worst analyses reusing scores |
| `analysis/gallery.html` | reproducible supportive/counterexample/random pairs with prompts, scores, IDs, model metadata and a human annotation CSV |
| `analysis/heatmaps/` | selected raw similarity arrays and overlays on actual processor geometry |
| `analysis/report.md` | generated methods, actual counts/results, limitations, links, `results_table.tex` and `paper_subsection.tex` |
| `analysis/discrimination_summary.*` | overall held-out concordance by ranking construction, with 95% intervals; `concept_effects.*` compares frozen discovery and test effects |
| `comparison/` | full-bank joined table, signed effect plot, rank correlation/overlap and frozen-list transfer for primary/best-rest constructions |

Score storage is `N_unique_images × 54,030 × 4` bytes: about 216 KB/image, 648 MB for 3,000 images, before metadata/thumbnails. The exact allocation is in `scores/cache.json`. Hash-identical decoded images share computation rows while legitimate statistical appearances stay in events. Scoring streams image batches and text chunks; analysis chunks concepts and bootstraps only selected candidates. Source ZIPs are read through bounded validated ranges and only required members persist. Images retain their original decoded formats, including WebP.

The method deliberately omits optional Residual-OMP, MLLM optimization/verification, masking heuristics, curated phrase additions and alternate encoders. No paid API is required. Null/synthetic fixtures are implementation checks, not empirical human-preference findings.
