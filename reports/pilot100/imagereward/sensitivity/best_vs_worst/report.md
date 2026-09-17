# Recorded human-preference pilot: imagereward

Status: held-out results available.

## Method

Frozen FG-CLIP normalized text/patch cosine similarity, maximum over real patches. The full supplied bank is the candidate set. Positive effects mean higher scores in preferred images. Scores are not probabilities of concept presence.

Construction: `best_vs_worst`. Equal weight per comparison within event, per event within normalized prompt, and per prompt. Matched preferred/rejected means have exactly the same difference as the weighted paired effect. Pair retention enables dependence-aware inference; pairing alone does not change that algebraic identity.

Discovery selects strictly positive/negative effects and freezes identity, direction and ordering before test aggregation. No held-out reselection. Whole incomplete or failed events are excluded. Splits preserve normalized prompts and known image/content dependencies; cross-official-split components are quarantined. Content-hash coverage is limited to resolved images.

Patch budget: 1024; vocabulary: 54030. The 1,024-patch pilot differs from the supplied fg_clip.py 16,384-patch setting. Frozen score configuration: `6b1345ec6e2c3189f7472dbe1f3d7007790922f7ea8b806f97fcbea4d446cef9`.

95% per-concept percentile cluster-bootstrap intervals use 2000 resamples, seed 42. Unequal-size components are resampled while preserving the prompt-weighted estimand. Fewer than two clusters gives unavailable intervals. These intervals are not simultaneous or tests of significance over the vocabulary.

## Actual observation counts

| Split | Prompts | Events | Comparisons | Components | Unique images |
|---|---:|---:|---:|---:|---:|
| discovery | 65 | 66 | 129 | 65 | 187 |
| validation | 15 | 15 | 27 | 15 | 40 |
| test | 15 | 15 | 27 | 15 | 40 |

## Frozen candidates in discovery order

| Concept | Side | Discovery Δ | Held-out Δ [95% CI] | Raw paired concordance |
|---|---|---:|---|---:|
| emerald-green | preferred | 0.0101 | -0.0054 [-0.0160, 0.0037] | 0.4833 |
| red-velvet | preferred | 0.0092 | -0.0060 [-0.0166, 0.0051] | 0.4500 |
| fierce-looking | preferred | 0.0088 | 0.0018 [-0.0027, 0.0067] | 0.5500 |
| hot-pink | preferred | 0.0088 | 0.0094 [-0.0162, 0.0326] | 0.6667 |
| big-eared | preferred | 0.0088 | 0.0006 [-0.0100, 0.0147] | 0.3500 |
| neon-green | preferred | 0.0087 | -0.0077 [-0.0319, 0.0109] | 0.5167 |
| building | preferred | 0.0081 | -0.0016 [-0.0136, 0.0093] | 0.4167 |
| red-rock | preferred | 0.0080 | -0.0135 [-0.0392, 0.0102] | 0.4167 |
| red-and-yellow | preferred | 0.0079 | -0.0099 [-0.0367, 0.0152] | 0.3000 |
| magenta | preferred | 0.0079 | 0.0007 [-0.0118, 0.0144] | 0.4667 |
| drainboard | rejected | -0.0118 | -0.0056 [-0.0168, 0.0054] | 0.4667 |
| bucket | rejected | -0.0118 | -0.0062 [-0.0217, 0.0075] | 0.5000 |
| bowl-shaped | rejected | -0.0117 | 0.0027 [-0.0113, 0.0182] | 0.4833 |
| yarmulke | rejected | -0.0114 | -0.0048 [-0.0206, 0.0117] | 0.4333 |
| clover | rejected | -0.0114 | 0.0053 [-0.0038, 0.0136] | 0.5833 |
| canister | rejected | -0.0112 | -0.0073 [-0.0263, 0.0105] | 0.4167 |
| crock | rejected | -0.0109 | -0.0038 [-0.0128, 0.0043] | 0.4167 |
| yogurt | rejected | -0.0107 | 0.0025 [-0.0058, 0.0104] | 0.6667 |
| cup-shaped | rejected | -0.0105 | -0.0004 [-0.0097, 0.0110] | 0.4333 |
| bread-and-butter | rejected | -0.0105 | -0.0005 [-0.0151, 0.0124] | 0.5000 |

## Limits and artifacts

Small pilots measure feasibility, not a publication-ready sample size. Intervals can be unstable with few components. Concept detections and galleries remain **unverified by humans**. Supportive extremes, counterexamples and a random audit are shown separately. Concepts can encode generator, resolution, prompt or user effects; matching does not identify why annotators chose images. A failed held-out effect is retained as a result.

Bank generation metadata is absent. Inspect `bank_audit.json` and `text_mode_audit.json` in the score directory before interpreting dense/text compatibility. Model/source revisions, vocabulary/embedding hashes, dtype, processor and score settings are recorded in `cache.json`. No MLLM verification, Residual-OMP, masking extension, reward training, or SigLIP baseline is claimed.

[All signed concept effects](all_concepts.csv), [frozen candidates](frozen_candidates.json), [held-out metrics](top_concepts.json), [dataset audit](dataset_audit.json), [paired gallery](gallery.html), [annotation template](annotation_template.csv), [confounder audit](confounder_audit.json), [stratified sensitivities](stratified_sensitivity.json), [LaTeX table](results_table.tex).

Inspect the separate ImageReward `sensitivity/best_vs_rest` and `sensitivity/best_vs_worst` analyses when available. Model/patch-count stratifications use frozen primary candidates. Annotator-component sensitivity preserves prompt weights and may have too few clusters for intervals.

## Measured text-bank compatibility

Sample re-encoding cosine agreement with the supplied bank: box mean=0.6770, minimum=0.4235, short mean=0.3944, minimum=0.2389, long mean=0.3626, minimum=0.2034.
These measurements do not establish the original encoding recipe. If agreement is weak, treat concept interpretations as provisional until the bank's model/template provenance is confirmed; dimensional compatibility alone is insufficient.

## Joint discovery-selected concept score

Equal mean of discovery-direction-adjusted concept scores; no fitted reward model. Values below use the same prompt-balanced test observations.

| Metric | Estimate | 95% component-bootstrap interval |
|---|---:|---|
| Preferred-minus-rejected similarity | -0.0007 | [-0.0086, 0.0058] |
| Paired concordance | 0.4333 | [0.2000, 0.6667] |

Concordance 0.5 is the equal-ordering reference; 0.6 means 60% weighted ordering credit, counting ties as half. An effect of +0.02 means +0.02 cosine-similarity units, not two percentage points of concept prevalence. Confidence intervals describe sampling uncertainty conditional on this scorer and cohort; they do not verify concept presence.
