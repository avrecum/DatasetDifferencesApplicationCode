> Compact export. Full tables and raw data are indexed in [artifact_index.json](artifact_index.json) and remain at `/e/project1/taco-vlm/avrecum/DatasetDifferencesApplicationsCode/runs/smoke/imagereward/analysis`. Relative full-table links refer to that original directory.

# Recorded human-preference pilot: imagereward

Status: held-out results available.

## Method

Frozen FG-CLIP normalized text/patch cosine similarity, maximum over real patches. The full supplied bank is the candidate set. Positive effects mean higher scores in preferred images. Scores are not probabilities of concept presence.

Construction: `all_strict_pairs`. Equal weight per comparison within event, per event within normalized prompt, and per prompt. Matched preferred/rejected means have exactly the same difference as the weighted paired effect. Pair retention enables dependence-aware inference; pairing alone does not change that algebraic identity.

Discovery selects strictly positive/negative effects and freezes identity, direction and ordering before test aggregation. No held-out reselection. Whole incomplete or failed events are excluded. Splits preserve normalized prompts and known image/content dependencies; cross-official-split components are quarantined. Content-hash coverage is limited to resolved images.

Patch budget: 1024; vocabulary: 54030. The 1,024-patch pilot differs from the supplied fg_clip.py 16,384-patch setting. Frozen score configuration: `d6c2cc1fbd950931a424a143baf3163424de5a8193230238dd96c5ce1e9180c5`.

95% per-concept percentile cluster-bootstrap intervals use 2000 resamples, seed 42. Unequal-size components are resampled while preserving the prompt-weighted estimand. Fewer than two clusters gives unavailable intervals. These intervals are not simultaneous or tests of significance over the vocabulary.

## Actual observation counts

| Split | Prompts | Events | Comparisons | Components | Unique images |
|---|---:|---:|---:|---:|---:|
| discovery | 5 | 5 | 92 | 5 | 33 |
| validation | 2 | 2 | 11 | 2 | 8 |
| test | 2 | 2 | 23 | 2 | 11 |

## Frozen candidates in discovery order

| Concept | Side | Discovery Δ | Held-out Δ [95% CI] | Raw paired concordance |
|---|---|---:|---|---:|
| red-and-green | preferred | 0.0253 | -0.0101 [-0.0268, 0.0066] | 0.3500 |
| red-and-yellow | preferred | 0.0251 | -0.0003 [-0.0187, 0.0180] | 0.5333 |
| red-letter | preferred | 0.0251 | -0.0222 [-0.0301, -0.0144] | 0.1944 |
| red-orange | preferred | 0.0232 | -0.0046 [-0.0107, 0.0014] | 0.4500 |
| red-tile | preferred | 0.0231 | -0.0210 [-0.0329, -0.0091] | 0.2222 |
| red-and-blue | preferred | 0.0218 | -0.0091 [-0.0187, 0.0005] | 0.3222 |
| red-blue | preferred | 0.0217 | -0.0152 [-0.0260, -0.0043] | 0.2944 |
| ruby-red | preferred | 0.0213 | -0.0002 [-0.0126, 0.0121] | 0.5944 |
| redware | preferred | 0.0205 | -0.0107 [-0.0203, -0.0011] | 0.3778 |
| red-meat | preferred | 0.0205 | -0.0174 [-0.0183, -0.0164] | 0.1667 |
| fencepost | rejected | -0.0266 | -0.0184 [-0.0353, -0.0014] | 0.2944 |
| stanchion | rejected | -0.0253 | -0.0297 [-0.0629, 0.0035] | 0.3056 |
| sailplane | rejected | -0.0234 | -0.0180 [-0.0491, 0.0130] | 0.4333 |
| flagpole | rejected | -0.0218 | -0.0008 [-0.0247, 0.0232] | 0.3333 |
| standpipe | rejected | -0.0209 | -0.0087 [-0.0216, 0.0041] | 0.4333 |
| peg | rejected | -0.0208 | -0.0106 [-0.0298, 0.0086] | 0.4056 |
| bollard | rejected | -0.0207 | -0.0280 [-0.0543, -0.0017] | 0.1389 |
| monorail | rejected | -0.0201 | -0.0028 [-0.0203, 0.0148] | 0.3889 |
| airfoil | rejected | -0.0199 | -0.0098 [-0.0160, -0.0036] | 0.3222 |
| nozzle | rejected | -0.0198 | -0.0093 [-0.0115, -0.0071] | 0.3667 |

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
| Preferred-minus-rejected similarity | 0.0013 | [-0.0042, 0.0068] |
| Paired concordance | 0.6944 | [0.3889, 1.0000] |

Concordance 0.5 is the equal-ordering reference; 0.6 means 60% weighted ordering credit, counting ties as half. An effect of +0.02 means +0.02 cosine-similarity units, not two percentage points of concept prevalence. Confidence intervals describe sampling uncertainty conditional on this scorer and cohort; they do not verify concept presence.
