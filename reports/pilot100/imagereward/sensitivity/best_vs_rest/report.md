# Recorded human-preference pilot: imagereward

Status: held-out results available.

## Method

Frozen FG-CLIP normalized text/patch cosine similarity, maximum over real patches. The full supplied bank is the candidate set. Positive effects mean higher scores in preferred images. Scores are not probabilities of concept presence.

Construction: `best_vs_rest`. Equal weight per comparison within event, per event within normalized prompt, and per prompt. Matched preferred/rejected means have exactly the same difference as the weighted paired effect. Pair retention enables dependence-aware inference; pairing alone does not change that algebraic identity.

Discovery selects strictly positive/negative effects and freezes identity, direction and ordering before test aggregation. No held-out reselection. Whole incomplete or failed events are excluded. Splits preserve normalized prompts and known image/content dependencies; cross-official-split components are quarantined. Content-hash coverage is limited to resolved images.

Patch budget: 1024; vocabulary: 54030. The 1,024-patch pilot differs from the supplied fg_clip.py 16,384-patch setting. Frozen score configuration: `6b1345ec6e2c3189f7472dbe1f3d7007790922f7ea8b806f97fcbea4d446cef9`.

95% per-concept percentile cluster-bootstrap intervals use 2000 resamples, seed 42. Unequal-size components are resampled while preserving the prompt-weighted estimand. Fewer than two clusters gives unavailable intervals. These intervals are not simultaneous or tests of significance over the vocabulary.

## Actual observation counts

| Split | Prompts | Events | Comparisons | Components | Unique images |
|---|---:|---:|---:|---:|---:|
| discovery | 65 | 66 | 494 | 65 | 408 |
| validation | 15 | 15 | 96 | 15 | 90 |
| test | 15 | 15 | 80 | 15 | 78 |

## Frozen candidates in discovery order

| Concept | Side | Discovery Δ | Held-out Δ [95% CI] | Raw paired concordance |
|---|---|---:|---|---:|
| green-painted | preferred | 0.0110 | -0.0003 [-0.0137, 0.0111] | 0.5589 |
| green-and-yellow | preferred | 0.0109 | -0.0063 [-0.0289, 0.0129] | 0.4500 |
| emerald-green | preferred | 0.0103 | -0.0081 [-0.0179, 0.0013] | 0.4011 |
| green-gold | preferred | 0.0095 | -0.0096 [-0.0248, 0.0034] | 0.3944 |
| neon-green | preferred | 0.0094 | -0.0051 [-0.0236, 0.0125] | 0.4678 |
| bright-green | preferred | 0.0090 | -0.0056 [-0.0189, 0.0058] | 0.4267 |
| green-black | preferred | 0.0088 | -0.0122 [-0.0275, 0.0026] | 0.4267 |
| green | preferred | 0.0088 | -0.0049 [-0.0148, 0.0036] | 0.4467 |
| moss-green | preferred | 0.0085 | -0.0016 [-0.0089, 0.0054] | 0.4978 |
| green-striped | preferred | 0.0085 | -0.0060 [-0.0230, 0.0074] | 0.3833 |
| wire-mesh | rejected | -0.0089 | -0.0062 [-0.0228, 0.0085] | 0.5011 |
| hail | rejected | -0.0077 | -0.0015 [-0.0114, 0.0086] | 0.4000 |
| bench-top | rejected | -0.0076 | 0.0005 [-0.0082, 0.0094] | 0.5000 |
| crock | rejected | -0.0074 | -0.0069 [-0.0127, -0.0020] | 0.3667 |
| bedrail | rejected | -0.0073 | 0.0085 [-0.0039, 0.0197] | 0.6300 |
| vise-like | rejected | -0.0072 | 0.0005 [-0.0084, 0.0091] | 0.5178 |
| vise | rejected | -0.0072 | 0.0002 [-0.0110, 0.0108] | 0.5600 |
| dustpan | rejected | -0.0072 | -0.0027 [-0.0141, 0.0098] | 0.4744 |
| scooper | rejected | -0.0071 | -0.0030 [-0.0117, 0.0073] | 0.3222 |
| locker | rejected | -0.0070 | 0.0059 [-0.0032, 0.0149] | 0.5856 |

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
| Preferred-minus-rejected similarity | -0.0027 | [-0.0102, 0.0033] |
| Paired concordance | 0.4356 | [0.3100, 0.5622] |

Concordance 0.5 is the equal-ordering reference; 0.6 means 60% weighted ordering credit, counting ties as half. An effect of +0.02 means +0.02 cosine-similarity units, not two percentage points of concept prevalence. Confidence intervals describe sampling uncertainty conditional on this scorer and cohort; they do not verify concept presence.
