# hard_diar.py — Sortformer vs ground truth on hard meeting

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from sortformer import load_model, diarize

PROB_THRESHOLD = 0.5

GROUND_TRUTH = [
    (0.0, 3.0, [0], "spk0 solo"),
    (3.0, 6.0, [1], "spk1 solo"),
    (6.0, 8.0, [0, 1], "crosstalk"),
    (8.0, 9.0, [], "silence"),
    (9.0, 10.5, [0], "spk0 rapid"),
    (10.5, 12.0, [1], "spk1 rapid"),
    (12.0, 13.5, [0], "spk0 rapid"),
    (13.5, 15.0, [1], "spk1 rapid"),
    (15.0, 16.0, [], "silence"),
    (16.0, 20.0, [2], "spk2 enters"),
    (20.0, 23.0, [0], "spk0 returns"),
    (23.0, 25.5, [0, 1, 2], "three-way"),
    (25.5, 28.0, [2], "spk2 finishes"),
    (28.0, 30.0, [1], "spk1 closing"),
    (30.0, 35.0, [], "silence"),
]


def analyze_region(probs, start_sec, end_sec, expected_spks, desc):
    frame_start = int(start_sec / 0.08)
    frame_end = min(int(end_sec / 0.08), probs.shape[0])

    if frame_start >= frame_end:
        return {"desc": desc, "match": False, "detail": "no frames"}

    region_probs = probs[frame_start:frame_end]
    avg_probs = region_probs.mean(axis=0)
    S = probs.shape[1]
    detected = [s for s in range(S) if avg_probs[s] > PROB_THRESHOLD]
    match = set(detected) == set(expected_spks)

    n_overlap_frames = sum(
        1 for f in range(region_probs.shape[0])
        if sum(region_probs[f] > PROB_THRESHOLD) >= 2
    )
    overlap_sec = n_overlap_frames * 0.08

    return {
        "desc": desc,
        "expected": expected_spks,
        "detected": detected,
        "match": match,
        "avg_probs": avg_probs,
        "overlap_sec": overlap_sec,
    }


def format_results(results, n_correct, n_total):
    lines = [
        f"{'REGION':>20}  {'EXPECTED':>10}  {'DETECTED':>10}  {'MATCH':>6}  {'OVERLAP':>8}",
        "=" * 70,
    ]
    for r in results:
        exp_str = ",".join(f"s{s}" for s in r["expected"]) if r["expected"] else "---"
        det_str = ",".join(f"s{s}" for s in r["detected"]) if r["detected"] else "---"
        match_str = "OK" if r["match"] else "MISS"
        overlap_str = f"{r['overlap_sec']:.1f}s" if r["overlap_sec"] > 0 else "---"
        lines.append(f"{r['desc']:>20}  {exp_str:>10}  {det_str:>10}  {match_str:>6}  {overlap_str:>8}")
        if not r["match"] and "avg_probs" in r:
            S = len(r["avg_probs"])
            probs_str = " ".join(f"s{i}={r['avg_probs'][i]:.2f}" for i in range(S))
            lines.append(f"{'':>20}  avg probs: {probs_str}")
    lines.extend(["=" * 70, f"Accuracy: {n_correct}/{n_total} ({100 * n_correct / n_total:.0f}%)"])
    return "\n".join(lines)


model = load_model()
segments, probs = diarize(model, config.HARD_AUDIO)
probs = probs.cpu().numpy() if isinstance(probs, torch.Tensor) else probs
_, T, S = probs.shape
probs = probs[0]

results = []
n_correct = 0
for start, end, spks, desc in GROUND_TRUTH:
    r = analyze_region(probs, start, end, spks, desc)
    results.append(r)
    if r["match"]:
        n_correct += 1

print(format_results(results, n_correct, len(GROUND_TRUTH)))
