"""Compare actual merged multi-GPU scores against previously saved single-GPU scores."""

import argparse
import numpy as np
from preference_diff.scoring import open_scores
from preference_diff.io import write_json

p = argparse.ArgumentParser(__doc__)
p.add_argument("--reference", required=True)
p.add_argument("--candidate", required=True)
p.add_argument("--output", required=True)
p.add_argument("--allow-batch-chunk-change", action="store_true")
p.add_argument("--allow-matching-invalid", action="store_true")
args = p.parse_args()
a, ai, av, vocab, am, _ = open_scores(args.reference)
b, bi, bv, bvocab, bm, _ = open_scores(args.candidate)
ac, bc = dict(am["configuration"]), dict(bm["configuration"])
if args.allow_batch_chunk_change:
    for key in ("image_batch_size", "text_chunk_size"):
        ac.pop(key)
        bc.pop(key)
assert vocab == bvocab and ac == bc, "Incompatible scientific score configurations"
if not args.allow_matching_invalid:
    assert bv.all(), "Multi-GPU test contains failed or uncommitted rows"
errors = []
invalid = 0
for iid, row in bi.items():
    assert iid in ai
    assert bv[row] == av[ai[iid]], "Validity differs between executions"
    if not bv[row]:
        invalid += 1
        continue
    errors.append(float(np.max(np.abs(a[ai[iid]] - b[row]))))
assert errors, "No common scored images"
maximum = max(errors)
passed = maximum <= 2e-4
write_json(
    args.output,
    dict(
        passed=passed,
        maximum_absolute_error=maximum,
        images=len(errors),
        matching_invalid_image_ids=invalid,
        reference_execution={
            k: am["configuration"][k] for k in ("image_batch_size", "text_chunk_size")
        },
        candidate_execution={
            k: bm["configuration"][k] for k in ("image_batch_size", "text_chunk_size")
        },
        concepts=len(vocab),
        absolute_tolerance=2e-4,
        reference_fingerprint=am["fingerprint"],
        candidate_fingerprint=bm["fingerprint"],
        scope="All supplied concepts for complete real-image groups; independent four-GPU recomputation",
    ),
)
assert passed, maximum
print(
    f"Four-GPU equivalence passed: {len(errors)} images × {len(vocab)} concepts; max error {maximum:.3g}"
)
