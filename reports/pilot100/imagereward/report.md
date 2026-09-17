> Compact export. Full tables and raw data are indexed in [artifact_index.json](artifact_index.json) and remain at `/e/project1/taco-vlm/avrecum/DatasetDifferencesApplicationsCode/runs/pilot100/imagereward/analysis`. Relative full-table links refer to that original directory.

# Recorded human-preference pilot: imagereward

Status: held-out results available.

## Method

Frozen FG-CLIP normalized text/patch cosine similarity, maximum over real patches. The full supplied bank is the candidate set. Positive effects mean higher scores in preferred images. Scores are not probabilities of concept presence.

Construction: `all_strict_pairs`. Equal weight per comparison within event, per event within normalized prompt, and per prompt. Matched preferred/rejected means have exactly the same difference as the weighted paired effect. Pair retention enables dependence-aware inference; pairing alone does not change that algebraic identity.

Discovery selects strictly positive/negative effects and freezes identity, direction and ordering before test aggregation. No held-out reselection. Whole incomplete or failed events are excluded. Splits preserve normalized prompts and known image/content dependencies; cross-official-split components are quarantined. Content-hash coverage is limited to resolved images.

Patch budget: 1024; vocabulary: 54030. The 1,024-patch pilot differs from the supplied fg_clip.py 16,384-patch setting. Frozen score configuration: `6b1345ec6e2c3189f7472dbe1f3d7007790922f7ea8b806f97fcbea4d446cef9`.

95% per-concept percentile cluster-bootstrap intervals use 2000 resamples, seed 42. Unequal-size components are resampled while preserving the prompt-weighted estimand. Fewer than two clusters gives unavailable intervals. These intervals are not simultaneous or tests of significance over the vocabulary.

## Actual observation counts

| Split | Prompts | Events | Comparisons | Components | Unique images |
|---|---:|---:|---:|---:|---:|
| discovery | 65 | 66 | 1005 | 65 | 408 |
| validation | 15 | 15 | 222 | 15 | 90 |
| test | 15 | 15 | 158 | 15 | 78 |

## Frozen candidates in discovery order

| Concept | Side | Discovery Δ | Held-out Δ [95% CI] | Raw paired concordance |
|---|---|---:|---|---:|
| emerald-green | preferred | 0.0062 | -0.0017 [-0.0077, 0.0037] | 0.4393 |
| red-flag | preferred | 0.0059 | -0.0115 [-0.0264, 0.0023] | 0.4123 |
| red-velvet | preferred | 0.0058 | -0.0063 [-0.0124, -0.0003] | 0.4395 |
| hot-pink | preferred | 0.0056 | 0.0057 [-0.0064, 0.0180] | 0.5638 |
| red-and-yellow | preferred | 0.0052 | -0.0062 [-0.0177, 0.0049] | 0.4534 |
| hairstyle | preferred | 0.0052 | 0.0022 [-0.0048, 0.0110] | 0.4732 |
| redcoat | preferred | 0.0051 | -0.0052 [-0.0150, 0.0039] | 0.4416 |
| lip-syncing | preferred | 0.0049 | 0.0043 [-0.0009, 0.0103] | 0.5313 |
| red-tile | preferred | 0.0048 | -0.0125 [-0.0280, 0.0003] | 0.4190 |
| neon-green | preferred | 0.0046 | -0.0006 [-0.0144, 0.0113] | 0.4798 |
| drainboard | rejected | -0.0071 | -0.0032 [-0.0099, 0.0037] | 0.4734 |
| chickweed | rejected | -0.0067 | 0.0001 [-0.0069, 0.0055] | 0.5281 |
| clover | rejected | -0.0064 | 0.0024 [-0.0024, 0.0071] | 0.5921 |
| spurge | rejected | -0.0063 | 0.0021 [-0.0031, 0.0086] | 0.5384 |
| bowl-shaped | rejected | -0.0063 | 0.0019 [-0.0054, 0.0098] | 0.5263 |
| hail | rejected | -0.0063 | 0.0008 [-0.0051, 0.0068] | 0.4933 |
| canister | rejected | -0.0063 | -0.0039 [-0.0139, 0.0061] | 0.4310 |
| weed | rejected | -0.0062 | 0.0059 [-0.0005, 0.0123] | 0.5771 |
| crock | rejected | -0.0062 | -0.0013 [-0.0074, 0.0036] | 0.4961 |
| yarmulke | rejected | -0.0059 | -0.0043 [-0.0134, 0.0042] | 0.4237 |

## Limits and artifacts

Small pilots measure feasibility, not a publication-ready sample size. Intervals can be unstable with few components. Concept detections and galleries remain **unverified by humans**. Supportive extremes, counterexamples and a random audit are shown separately. Concepts can encode generator, resolution, prompt or user effects; matching does not identify why annotators chose images. A failed held-out effect is retained as a result.

Bank generation metadata is absent. Inspect `bank_audit.json` and `text_mode_audit.json` in the score directory before interpreting dense/text compatibility. Model/source revisions, vocabulary/embedding hashes, dtype, processor and score settings are recorded in `cache.json`. No MLLM verification, Residual-OMP, masking extension, reward training, or SigLIP baseline is claimed.

[All signed concept effects](all_concepts.csv), [frozen candidates](frozen_candidates.json), [held-out metrics](top_concepts.json), [dataset audit](dataset_audit.json), [paired gallery](gallery.html), [annotation template](annotation_template.csv), [confounder audit](confounder_audit.json), [stratified sensitivities](stratified_sensitivity.json), [LaTeX table](results_table.tex).

[Overall discrimination plot](discrimination_summary.png), [discrimination table](discrimination_summary.csv), and [discovery-versus-held-out concept effects](concept_effects.png). The overall score combines concepts frozen on discovery; 0.5 is the paired equal-ordering reference.

Inspect the separate ImageReward `sensitivity/best_vs_rest` and `sensitivity/best_vs_worst` analyses when available. Model/patch-count stratifications use frozen primary candidates. Annotator-component sensitivity preserves prompt weights and may have too few clusters for intervals.

## Measured text-bank compatibility

Sample re-encoding cosine agreement with the supplied bank: box mean=0.6770, minimum=0.4235, short mean=0.3944, minimum=0.2389, long mean=0.3626, minimum=0.2034.
These measurements do not establish the original encoding recipe. If agreement is weak, treat concept interpretations as provisional until the bank's model/template provenance is confirmed; dimensional compatibility alone is insufficient.

## Joint discovery-selected concept score

Equal mean of discovery-direction-adjusted concept scores; no fitted reward model. Values below use the same prompt-balanced test observations.

| Metric | Estimate | 95% component-bootstrap interval |
|---|---:|---|
| Preferred-minus-rejected similarity | -0.0016 | [-0.0058, 0.0024] |
| Paired concordance | 0.4526 | [0.3521, 0.5674] |

Concordance 0.5 is the equal-ordering reference; 0.6 means 60% weighted ordering credit, counting ties as half. An effect of +0.02 means +0.02 cosine-similarity units, not two percentage points of concept prevalence. Confidence intervals describe sampling uncertainty conditional on this scorer and cohort; they do not verify concept presence.

## Extended text-bank diagnostic

An independently run diagnostic tested 66 fixed mode/template/mask combinations on 32 vocabulary entries, without preference labels. Closest was `box` with template `a photo of a {}.`: mean cosine **0.9656**, minimum **0.8721**, same-row nearest match fraction **1.000** over the full bank.
This supports compatibility with templated dense-text embeddings more strongly than the raw-word check. It does not identify the exact original generation recipe, establish detector accuracy, or replace the bank. [Complete diagnostic](bank_template_diagnostic.json).
