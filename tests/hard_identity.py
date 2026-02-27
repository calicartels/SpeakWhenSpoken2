# hard_identity.py — ECAPA on clean source audio (pre-mix)

import os
import sys

import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from INPUT_PIPELINE.ecapa import load_model


def extract(model, path):
    wav, sr = torchaudio.load(path)
    if sr != config.SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, config.SAMPLE_RATE)
    emb = model.encode_batch(wav).squeeze().cpu()
    return emb


def cosine(a, b):
    return torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)


SEGMENTS = [
    "spk0_early", "spk0_rapid", "spk0_return",
    "spk1_early", "spk1_rapid", "spk1_close",
    "spk2_solo", "spk2_late",
]

SAME_SPEAKER = [
    ("spk0_early", "spk0_rapid", "spk0 early vs rapid"),
    ("spk0_early", "spk0_return", "spk0 early vs return"),
    ("spk0_rapid", "spk0_return", "spk0 rapid vs return"),
    ("spk1_early", "spk1_rapid", "spk1 early vs rapid"),
    ("spk1_early", "spk1_close", "spk1 early vs closing"),
    ("spk1_rapid", "spk1_close", "spk1 rapid vs closing"),
    ("spk2_solo", "spk2_late", "spk2 first vs later"),
]

CROSS_SPEAKER = [
    ("spk0_early", "spk1_early", "spk0 vs spk1"),
    ("spk0_early", "spk2_solo", "spk0 vs spk2"),
    ("spk1_early", "spk2_solo", "spk1 vs spk2"),
]


def verdict(score, expect_same):
    if expect_same:
        if score > 0.85:
            return "SAME (high)"
        if score > 0.60:
            return "SAME (likely)"
        if score > 0.40:
            return "UNCERTAIN"
        return "FAIL"
    if score < 0.40:
        return "DIFFERENT"
    if score < 0.60:
        return "DIFFERENT (weak)"
    return "FAIL"


def format_results(embs):
    lines = ["SAME SPEAKER (expect >0.85)", f"{'TEST':>30}  {'SCORE':>6}  VERDICT", "-" * 55]
    same_pass = 0
    for a, b, desc in SAME_SPEAKER:
        s = cosine(embs[a], embs[b]).item()
        v = verdict(s, expect_same=True)
        if s > 0.60:
            same_pass += 1
        lines.append(f"{desc:>30}  {s:6.3f}  {v}")
    lines.extend(["", "CROSS SPEAKER (expect <0.60)", f"{'TEST':>30}  {'SCORE':>6}  VERDICT", "-" * 55])
    cross_pass = 0
    for a, b, desc in CROSS_SPEAKER:
        s = cosine(embs[a], embs[b]).item()
        v = verdict(s, expect_same=False)
        if s < 0.60:
            cross_pass += 1
        lines.append(f"{desc:>30}  {s:6.3f}  {v}")
    total = len(SAME_SPEAKER) + len(CROSS_SPEAKER)
    passed = same_pass + cross_pass
    lines.append(f"\n{passed}/{total} passed ({100 * passed / total:.0f}%)")
    return "\n".join(lines)


model = load_model()
embs = {}
for name in SEGMENTS:
    path = os.path.join(config.CLEAN_DIR, f"{name}.wav")
    if not os.path.exists(path):
        print(f"Missing {path}, run tests/hard_audio.py first")
        raise SystemExit(1)
    embs[name] = extract(model, path)

print(format_results(embs))
