import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
import gate

FRAME_SEC = 0.08
VAP_EVERY_N = 4


def compute_baseline(probs_sequence):
    n_frames = len(probs_sequence)
    raw = []

    for i in range(0, n_frames, VAP_EVERY_N):
        ts = i * FRAME_SEC
        probs = probs_sequence[i]
        max_prob = max(probs)
        score = 1.0 - max_prob

        active = [j for j, p in enumerate(probs) if p > 0.3]
        mode = "silence" if not active else ("solo" if len(active) == 1 else "dyad")
        dom_idx = int(np.argmax(probs))
        dom_prob = probs[dom_idx] if max_prob > 0.3 else None

        if gate.should_open(score, mode, dom_prob):
            raw.append({
                "timestamp": ts,
                "ai_opening": score,
                "active_speakers": active,
                "mode": mode,
                "dominant_prob": dom_prob,
            })

    return raw, gate.filter_openings(raw)


def compare_at_timestamps(probs_sequence, vap_openings):
    lines = [f"{'Time':>8}  {'VAP':>6}  {'Base':>6}  {'Reason':>18}  Winner", "-" * 60]
    for ho in vap_openings:
        ts = ho["timestamp"]
        frame = int(ts / FRAME_SEC)
        if frame >= len(probs_sequence):
            continue
        probs = probs_sequence[frame]
        base = 1.0 - max(probs)
        vap_score = ho["ai_opening"]
        reason = ho.get("reason", "unknown")
        winner = "Tie" if base >= config.GATE_THRESHOLD_SILENCE else "VAP"
        m, s = int(ts // 60), ts % 60
        lines.append(f"{m}:{s:04.1f}  {vap_score:6.2f}  {base:6.2f}  {reason:>18}  {winner}")
    return "\n".join(lines)


def format_summary(vap_openings, baseline_openings):
    vap_times = {round(o["timestamp"], 1) for o in vap_openings}
    base_times = {round(o["timestamp"], 1) for o in baseline_openings}

    vap_matched, base_matched = set(), set()
    for vt in sorted(vap_times):
        for bt in sorted(base_times):
            if abs(vt - bt) <= config.GATE_SUPPRESS_SEC and bt not in base_matched:
                vap_matched.add(vt)
                base_matched.add(bt)
                break

    lines = [
        f"VAP openings: {len(vap_openings)}",
        f"Baseline openings: {len(baseline_openings)}",
        f"Matched (within {config.GATE_SUPPRESS_SEC}s): {len(vap_matched)}",
        f"VAP-only: {len(vap_times) - len(vap_matched)}",
        f"Baseline-only: {len(base_times) - len(base_matched)}",
    ]

    vap_unique = [o for o in vap_openings if round(o["timestamp"], 1) not in vap_matched]
    if vap_unique:
        lines.append("\nVAP catches that baseline misses:")
        for o in vap_unique:
            m, s = int(o["timestamp"] // 60), o["timestamp"] % 60
            lines.append(f"  [{m}:{s:04.1f}] {o.get('reason', '?')} vap={o['ai_opening']:.2f}")

    base_unique = [o for o in baseline_openings if round(o["timestamp"], 1) not in base_matched]
    if base_unique:
        lines.append("\nBaseline catches that VAP misses:")
        for o in base_unique:
            m, s = int(o["timestamp"] // 60), o["timestamp"] % 60
            lines.append(f"  [{m}:{s:04.1f}] {o.get('reason', '?')} base={o['ai_opening']:.2f}")

    return "\n".join(lines)
