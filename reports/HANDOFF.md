# Implementation and actual execution handoff

The reusable application is implemented and tested. Real ImageReward scoring, held-out evaluation, sensitivities, galleries and patch maps completed on a Slurm GPU allocation. **This pilot does not demonstrate reliable preference discrimination.** Pick-a-Pic empirical results and cross-dataset replication remain unavailable because the original image URLs failed. The supplied text bank's exact encoder/template provenance is also unresolved.

## Read these artifacts first

- [ImageReward pilot report](pilot100/imagereward/report.md), [all twenty held-out candidates](pilot100/imagereward/top_concepts.json), [paired gallery](pilot100/imagereward/gallery.html), [human annotation template](pilot100/imagereward/annotation_template.csv), [patch maps](pilot100/imagereward/heatmaps/index.json).
- [Best-versus-rest report](pilot100/imagereward/sensitivity/best_vs_rest/report.md) and [best-versus-worst report](pilot100/imagereward/sensitivity/best_vs_worst/report.md).
- [LaTeX candidate table](pilot100/imagereward/results_table.tex), [proposed application subsection](human_preference_application.tex), [construction sensitivity table](pilot_results_table.tex).
- [Pick-a-Pic unavailable-result report](smoke/pickapic/report.md), [cross-dataset availability audit](pilot100/comparison/primary/comparison_summary.json), [source/schema inspection](source_inspection/source_inspection.json), [method/source review](../docs/METHOD_AND_SOURCE_REVIEW.md).
- [Machine-readable execution summary](execution_summary.json) and [full artifact locations and SHA256 hashes](pilot100/imagereward/artifact_index.json).

The full score matrix, row mapping, validity mask, manifests and 54,030-row signed effect tables remain in `runs/pilot100/imagereward/`. Compact review artifacts are committed here. The report export's artifact index identifies every full local file; data/model caches and the environment are Git-ignored.

## What actually ran

| Run | Actual execution |
|---|---|
| Offline tests | 25 passed; the GPU test was skipped on the CPU login node. Synthetic known-sign and null fixtures, leakage/weighting/cache tests, and an offline end-to-end report/comparison test ran. |
| Real GPU equivalence | The separately scheduled GPU test passed. Batched dense-head scores versus the official FP32 single-image API: maximum absolute error `1.1920928955078125e-07`, tolerance `atol=rtol=2e-4`, two real images with contrasting aspect ratios, 128 evenly spaced concepts. |
| Lint | Ruff checks passed for package, tests and artifact exporter. |
| Real smoke | Slurm job `1863367`, completed `0:0`, 5:12. ImageReward: 52 scored images, 5 discovery / 2 validation / 2 test prompts. Pick-a-Pic: 119 requested images, all HTTP 403, no valid images. |
| Real bounded pilot | Slurm job `1863492`, completed `0:0`, 5:40 including the GPU test, scoring, analysis and maps. Requested 100 distinct ImageReward prompts / 101 events / 607 images; 576 images decoded and scored, 31 failed decoding. Excluding all five affected events left 95 prompts / 96 events. No failed group was replenished. |

All real scores used the pinned FG-CLIP2 base model, the original **54,030 × 768** text bank, FP32 and a **1,024-patch budget**, seed 42, official single-image production inference, and 2,000 bootstrap resamples. The supplied script's 16,384-patch setting is exposed as a separate reproduction preset. The 100-prompt bound was an engineering pilot, not a power calculation; the CLI also supports 500-prompt and full runs. No GPU work ran on a login node.

The pilot's usable discovery/validation/test counts were **65/15/15 prompts**, **66/15/15 events**, and **1,005/222/158 strict comparisons**. The discovery duplicate prompt correctly shares total prompt weight across its two events. All 15 held-out components were distinct prompt components. Hash leakage checks passed for resolved images.

## Observed results and interpretation

The primary frozen twenty-concept score achieved **0.4526 paired concordance, 95% CI [0.3521, 0.5674]**, and a preferred-minus-rejected effect of **−0.001615**, CI **[−0.005756, +0.002446]**. Concordance 0.5 is the equal-ordering reference; these data do not establish performance above it. **7/20** concepts retained their discovery effect direction. That is descriptive direction replication, not a significance test or verification of concept presence.

The first preferred-side discovery terms were `emerald-green`, `red-flag`, and `red-velvet`; all three reversed sign on the held-out sample. They remain in the main table in discovery order. These are returned vocabulary concepts, not verified human preference attributes. No replacement concepts were selected from test data.

| Construction | Test comparisons | Frozen joint concordance [95% CI] |
|---|---:|---|
| All strict rank pairs | 158 | 0.4526 [0.3521, 0.5674] |
| Best versus rest | 80 | 0.4356 [0.3100, 0.5622] |
| Best versus worst | 27 | 0.4333 [0.2000, 0.6667] |

Each sensitivity has its own discovery-frozen candidate list and reuses the identical image scores. None of these intervals establishes above-reference discrimination. All twenty primary held-out effect intervals included zero or pointed opposite to the discovery direction; none supported its discovery direction with a per-concept interval entirely on that side.

A score difference of `+0.02` means 0.02 cosine-similarity units higher in preferred images under the stated hierarchical weights. It does **not** mean a two-percentage-point increase in concept prevalence. A concordance of `0.60` means 60% weighted ordering credit, counting ties as half. The raw concordance always asks whether preferred images score higher; the direction-adjusted version reverses that orientation only for concepts assigned to the rejected side on discovery data. Confidence intervals reflect sampling uncertainty conditional on the scorer/cohort, not causal evidence or detector accuracy.

Patch counts ranged from 968 to 1,024 across the 576 images (median 1,024). All 15 test events had equal patch counts and equal aspect ratios within the event, so those restricted sensitivities retain the primary test cohort. ImageReward generator and annotator IDs are unavailable; their effects cannot be separated here. Category sensitivities have very small groups. Galleries and maps remain human-unverified.

## Source discoveries and blockers

- **ImageReward:** 55,247 metadata images, 8,878 ranking groups and 136,892 strict comparisons before split quarantine, reproducing the published pair total. Lower rank is preferred and ties are skipped. Prompt/image dependencies crossing official split boundaries quarantine 154 events. At least one failed pilot image was a zero-byte member of the pinned archive; failures were not converted into zero scores.
- **Linked Pick-a-Pic:** 25,355 four-candidate ranking events and 109,356 image rows. Strict metadata eligibility leaves 13,085 events. Ordered exclusion counts: 7,189 invalid/none selections, 5,043 candidate-prompt mismatches, 37 unresolved metadata candidates, and one remaining incompatible negative-prompt event. There are 11,654 reused candidate image IDs; these do not become permanent preferred/rejected labels.
- **Image availability:** every one of the 119 selected Pick-a-Pic image URLs returned HTTP 403. The authors describe obsolete AWS URLs; four investigated alternate Hub endpoints returned HTTP 401. No UID-verifiable replacement bytes were obtained and no different pairwise release was substituted. Accessible copies of the original candidate UIDs are required to finish that empirical analysis.
- **Bank provenance:** twelve sampled entries re-encoded with documented modes had mean cosine agreement 0.6770 for `box`, 0.3944 for `short`, and 0.3626 for `long`; the minimum box agreement was 0.4235. Box was closest but did not closely reproduce the bank. Confirm the original encoder revision, template and text mode before making semantic claims or interpreting a larger run. The supplied bank was preserved.
- **Storage/environment:** scratch creation hit a quota error, so bounded runs use project storage. Compute nodes lack outbound network access; the pinned model is prefetched on the login node and loaded locally. The application uses an isolated Transformers layer over the existing numerical runtime; unrelated inherited package-metadata conflicts are documented in the source review. The base environment was not changed.

## Commands

From the repository root, reproduce the saved pilot's analysis without rerunning inference:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m preference_diff analyze --output runs/pilot100/imagereward --bootstrap 2000
.venv/bin/python -m preference_diff report --output runs/pilot100/imagereward
```

Run a fresh full experiment once the original Pick-a-Pic image cache is available. Replace the one local-image placeholder and choose storage with adequate quota. This is the exact 1,024-patch configuration used by the pilot; full execution was **not** performed here:

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

For an independently bounded ImageReward-only 500-prompt run:

```bash
export HF_HOME="$PROJECT/avrecum/.hf"
.venv/bin/python -m preference_diff prepare --dataset imagereward \
  --output runs/pilot500/imagereward --group-limit 500 --seed 42
sbatch --time=01:00:00 scripts/imagereward_pilot.sbatch runs/pilot500/imagereward
```

The README documents all subcommands, paths, resume safeguards, alternative patch settings and local metadata support. No paid API or MLLM access is needed.
