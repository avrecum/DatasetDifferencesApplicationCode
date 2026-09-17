"""Summarize measured utilization; startup/CPU phases remain visible in averages."""

import argparse
import csv
from collections import defaultdict
import numpy as np
from preference_diff.io import write_json

p = argparse.ArgumentParser(__doc__)
p.add_argument("csv_path")
p.add_argument("--output", required=True)
args = p.parse_args()
observations = defaultdict(list)
with open(args.csv_path) as f:
    for original in csv.DictReader(f):
        row = {
            k.strip(): v.strip() if v is not None else "" for k, v in original.items()
        }
        try:
            gpu = int(row["index"])
            value = float(row["utilization.gpu [%]"].split()[0])
            mem = float(row["memory.used [MiB]"].split()[0])
        except (KeyError, ValueError, IndexError):
            continue
        observations[gpu].append((value, mem))
results = []
for gpu, rows in sorted(observations.items()):
    values, memory = np.array(rows).T
    active = values[values > 0]
    results.append(
        dict(
            gpu_index=gpu,
            samples=len(values),
            peak_utilization_percent=float(values.max()),
            mean_utilization_percent=float(values.mean()),
            samples_with_gpu_work=len(active),
            active_sample_mean_percent=float(active.mean()) if len(active) else None,
            peak_memory_mib=float(memory.max()),
        )
    )
write_json(
    args.output,
    dict(
        source_csv=args.csv_path,
        gpus=results,
        all_four_observed_work=len(results) == 4
        and all(r["samples_with_gpu_work"] > 0 for r in results),
        interpretation="Two-second snapshots include initialization, CPU analysis and shutdown. Brief tests may be dominated by startup and sampled peaks can miss short kernels. This is measured utilization, not a claim of 100% efficiency.",
    ),
)
