# Recorded human-preference pilot: imagereward

Status: held-out results available.

## Method

Frozen FG-CLIP normalized text/patch cosine similarity, maximum over real patches. The full supplied bank is the candidate set. Positive effects mean higher scores in preferred images. Scores are not probabilities of concept presence.

Construction: `best_vs_rest`. Equal weight per comparison within event, per event within normalized prompt, and per prompt. Matched preferred/rejected means have exactly the same difference as the weighted paired effect. Pair retention enables dependence-aware inference; pairing alone does not change that algebraic identity.

Discovery selects strictly positive/negative effects and freezes identity, direction and ordering before test aggregation. No held-out reselection. Whole incomplete or failed events are excluded. Splits preserve normalized prompts and known image/content dependencies; cross-official-split components are quarantined. Content-hash coverage is limited to resolved images.

Patch budget: 1024; vocabulary: 54030. The 1,024-patch pilot differs from the supplied fg_clip.py 16,384-patch setting. Frozen score configuration: `d6c2cc1fbd950931a424a143baf3163424de5a8193230238dd96c5ce1e9180c5`.

95% per-concept percentile cluster-bootstrap intervals use 2000 resamples, seed 42. Unequal-size components are resampled while preserving the prompt-weighted estimand. Fewer than two clusters gives unavailable intervals. These intervals are not simultaneous or tests of significance over the vocabulary.

## Actual observation counts

| Split | Prompts | Events | Comparisons | Components | Unique images |
|---|---:|---:|---:|---:|---:|
| discovery | 5 | 5 | 46 | 5 | 33 |
| validation | 2 | 2 | 7 | 2 | 8 |
| test | 2 | 2 | 13 | 2 | 11 |

## Frozen candidates in discovery order

| Concept | Side | Discovery Δ | Held-out Δ [95% CI] | Raw paired concordance |
|---|---|---:|---|---:|
| red-and-green | preferred | 0.0330 | -0.0100 [-0.0505, 0.0305] | 0.3500 |
| blue-and-green | preferred | 0.0312 | -0.0036 [-0.0135, 0.0063] | 0.3167 |
| red-and-blue | preferred | 0.0309 | -0.0026 [-0.0129, 0.0078] | 0.4667 |
| red-blue | preferred | 0.0299 | -0.0088 [-0.0217, 0.0041] | 0.4167 |
| red-and-yellow | preferred | 0.0296 | -0.0110 [-0.0374, 0.0153] | 0.3000 |
| blue-striped | preferred | 0.0296 | -0.0058 [-0.0204, 0.0088] | 0.5167 |
| codex | preferred | 0.0294 | -0.0191 [-0.0439, 0.0057] | 0.4000 |
| red-letter | preferred | 0.0286 | -0.0058 [-0.0269, 0.0153] | 0.2500 |
| choker | preferred | 0.0285 | 0.0213 [-0.0076, 0.0502] | 0.6000 |
| redware | preferred | 0.0280 | -0.0181 [-0.0352, -0.0010] | 0.3000 |
| hanger-on | rejected | -0.0331 | -0.0005 [-0.0151, 0.0142] | 0.6667 |
| hanger | rejected | -0.0305 | -0.0101 [-0.0354, 0.0153] | 0.6167 |
| nozzle | rejected | -0.0299 | -0.0218 [-0.0340, -0.0095] | 0.1500 |
| guyline | rejected | -0.0294 | -0.0306 [-0.0677, 0.0064] | 0.2500 |
| sailplane | rejected | -0.0283 | -0.0279 [-0.0655, 0.0097] | 0.3000 |
| bollard | rejected | -0.0279 | -0.0313 [-0.0674, 0.0047] | 0.2500 |
| needle | rejected | -0.0274 | -0.0173 [-0.0339, -0.0007] | 0.2500 |
| monorail | rejected | -0.0270 | 0.0007 [-0.0162, 0.0176] | 0.4000 |
| stanchion | rejected | -0.0269 | -0.0392 [-0.0751, -0.0032] | 0.3000 |
| skewer | rejected | -0.0267 | -0.0074 [-0.0281, 0.0133] | 0.3500 |

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
| Preferred-minus-rejected similarity | 0.0061 | [-0.0001, 0.0123] |
| Paired concordance | 0.7500 | [0.5000, 1.0000] |

Concordance 0.5 is the equal-ordering reference; 0.6 means 60% weighted ordering credit, counting ties as half. An effect of +0.02 means +0.02 cosine-similarity units, not two percentage points of concept prevalence. Confidence intervals describe sampling uncertainty conditional on this scorer and cohort; they do not verify concept presence.
