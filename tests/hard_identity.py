# hard_identity.py — ECAPA same/different speaker on hard meeting

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from ecapa import load_model, load_audio


def extract(model, wav, start_sec, end_sec):
    s = int(start_sec * config.SAMPLE_RATE)
    e = int(end_sec * config.SAMPLE_RATE)
    chunk = wav[s:e].unsqueeze(0)
    return model.encode_batch(chunk).squeeze().cpu()


def cosine(a, b):
    return torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)


SEGMENTS = {
    "spk0_early": (0.5, 2.5),
    "spk1_early": (3.5, 5.5),
    "spk0_rapid": (9.0, 10.5),
    "spk1_rapid": (10.5, 12.0),
    "spk2_solo": (16.5, 19.5),
    "spk0_return": (20.5, 22.5),
    "spk2_late": (26.0, 27.5),
    "spk1_close": (28.0, 29.5),
}

TESTS = [
    ("spk0_early", "spk0_return", "spk0 early vs return"),
    ("spk0_early", "spk0_rapid", "spk0 early vs rapid"),
    ("spk1_early", "spk1_close", "spk1 early vs closing"),
    ("spk2_solo", "spk2_late", "spk2 first vs later"),
    ("spk0_early", "spk1_early", "spk0 vs spk1"),
    ("spk0_early", "spk2_solo", "spk0 vs spk2"),
    ("spk1_early", "spk2_solo", "spk1 vs spk2"),
]


def verdict(score):
    if score > 0.85:
        return "SAME (high)"
    if score > 0.60:
        return "SAME (likely)"
    if score > 0.40:
        return "UNCERTAIN"
    return "DIFFERENT"


def format_results(embs):
    lines = [f"{'TEST':>45}  {'SCORE':>6}  VERDICT", "-" * 75]
    for name_a, name_b, desc in TESTS:
        score = cosine(embs[name_a], embs[name_b]).item()
        lines.append(f"{desc:>45}  {score:6.3f}  {verdict(score)}")
    return "\n".join(lines)


model = load_model()
wav = load_audio(config.HARD_AUDIO)
embs = {name: extract(model, wav, s, e) for name, (s, e) in SEGMENTS.items()}
print(format_results(embs))
