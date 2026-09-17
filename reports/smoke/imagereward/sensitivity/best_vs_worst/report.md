# Recorded human-preference pilot: imagereward

Status: held-out results available.

## Method

Frozen FG-CLIP normalized text/patch cosine similarity, maximum over real patches. The full supplied bank is the candidate set. Positive effects mean higher scores in preferred images. Scores are not probabilities of concept presence.

Construction: `best_vs_worst`. Equal weight per comparison within event, per event within normalized prompt, and per prompt. Matched preferred/rejected means have exactly the same difference as the weighted paired effect. Pair retention enables dependence-aware inference; pairing alone does not change that algebraic identity.

Discovery selects strictly positive/negative effects and freezes identity, direction and ordering before test aggregation. No held-out reselection. Whole incomplete or failed events are excluded. Splits preserve normalized prompts and known image/content dependencies; cross-official-split components are quarantined. Content-hash coverage is limited to resolved images.

Patch budget: 1024; vocabulary: 54030. The 1,024-patch pilot differs from the supplied fg_clip.py 16,384-patch setting. Frozen score configuration: `d6c2cc1fbd950931a424a143baf3163424de5a8193230238dd96c5ce1e9180c5`.

95% per-concept percentile cluster-bootstrap intervals use 2000 resamples, seed 42. Unequal-size components are resampled while preserving the prompt-weighted estimand. Fewer than two clusters gives unavailable intervals. These intervals are not simultaneous or tests of significance over the vocabulary.

## Actual observation counts

| Split | Prompts | Events | Comparisons | Components | Unique images |
|---|---:|---:|---:|---:|---:|
| discovery | 5 | 5 | 10 | 5 | 15 |
| validation | 2 | 2 | 3 | 2 | 5 |
| test | 2 | 2 | 5 | 2 | 6 |

## Frozen candidates in discovery order

| Concept | Side | Discovery Δ | Held-out Δ [95% CI] | Raw paired concordance |
|---|---|---:|---|---:|
| red-orange | preferred | 0.0535 | -0.0105 [-0.0178, -0.0032] | 0.2500 |
| red-and-yellow | preferred | 0.0499 | -0.0017 [-0.0311, 0.0277] | 0.3750 |
| red-giant | preferred | 0.0496 | -0.0193 [-0.0212, -0.0174] | 0.2500 |
| red-rock | preferred | 0.0477 | -0.0337 [-0.0385, -0.0289] | 0.1250 |
| red-meat | preferred | 0.0467 | -0.0325 [-0.0345, -0.0305] | 0.2500 |
| tree-ring | preferred | 0.0461 | 0.0019 [0.0006, 0.0032] | 0.7500 |
| reddish-orange | preferred | 0.0447 | -0.0015 [-0.0054, 0.0025] | 0.2500 |
| red-and-green | preferred | 0.0435 | -0.0190 [-0.0447, 0.0067] | 0.2500 |
| red-tile | preferred | 0.0429 | -0.0349 [-0.0548, -0.0150] | 0.2500 |
| reddish-purple | preferred | 0.0400 | -0.0385 [-0.0440, -0.0329] | 0.0000 |
| stanchion | rejected | -0.0436 | -0.0505 [-0.1048, 0.0039] | 0.3750 |
| escalator | rejected | -0.0434 | -0.0354 [-0.0953, 0.0245] | 0.5000 |
| fencepost | rejected | -0.0433 | -0.0319 [-0.0588, -0.0049] | 0.0000 |
| peg | rejected | -0.0401 | -0.0168 [-0.0497, 0.0160] | 0.3750 |
| top-shelf | rejected | -0.0395 | -0.0143 [-0.0293, 0.0006] | 0.2500 |
| clothespin | rejected | -0.0380 | 0.0106 [-0.0133, 0.0345] | 0.5000 |
| nozzle | rejected | -0.0379 | -0.0172 [-0.0192, -0.0152] | 0.1250 |
| bollard | rejected | -0.0373 | -0.0461 [-0.0906, -0.0016] | 0.2500 |
| trash | rejected | -0.0370 | -0.0096 [-0.0358, 0.0165] | 0.5000 |
| cigarette | rejected | -0.0367 | 0.0079 [-0.0089, 0.0247] | 0.5000 |

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
| Preferred-minus-rejected similarity | 0.0007 | [-0.0103, 0.0117] |
| Paired concordance | 0.7500 | [0.5000, 1.0000] |

Concordance 0.5 is the equal-ordering reference; 0.6 means 60% weighted ordering credit, counting ties as half. An effect of +0.02 means +0.02 cosine-similarity units, not two percentage points of concept prevalence. Confidence intervals describe sampling uncertainty conditional on this scorer and cohort; they do not verify concept presence.
