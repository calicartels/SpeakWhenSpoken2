# hard_audio.py — 35s meeting with crosstalk, rapid turns, speaker return

import os
import sys

import numpy as np
import torch
import torchaudio
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def load_speakers(n=3):
    ds = load_dataset(
        "librispeech_asr", "clean",
        split="test",
        streaming=True,
    )
    speakers = []
    seen = set()
    for item in ds:
        spk = item["speaker_id"]
        if spk in seen:
            continue
        audio = np.array(item["audio"]["array"], dtype=np.float32)
        if len(audio) < 6 * config.SAMPLE_RATE:
            continue
        seen.add(spk)
        speakers.append(audio)
        if len(speakers) >= n:
            break
    return speakers


def place(mix, audio, start_sec, dur_sec, sr):
    s = int(start_sec * sr)
    n = int(dur_sec * sr)
    chunk = audio[:n]
    end = s + len(chunk)
    if end > len(mix):
        chunk = chunk[: len(mix) - s]
        end = s + len(chunk)
    mix[s:end] += chunk


def build_hard_meeting(speakers):
    sr = config.SAMPLE_RATE
    total_sec = 35
    mix = np.zeros(total_sec * sr, dtype=np.float32)
    a, b, c = speakers[0], speakers[1], speakers[2]

    place(mix, a[0:], 0.0, 3.0, sr)
    place(mix, b[0:], 3.0, 3.0, sr)
    place(mix, a[3 * sr :], 6.0, 2.0, sr)
    place(mix, b[3 * sr :], 6.0, 2.0, sr)
    place(mix, a[5 * sr :], 9.0, 1.5, sr)
    place(mix, b[5 * sr :], 10.5, 1.5, sr)
    place(mix, a[int(6.5 * sr) :], 12.0, 1.5, sr)
    place(mix, b[int(6.5 * sr) :], 13.5, 1.5, sr)
    place(mix, c[0:], 16.0, 4.0, sr)
    place(mix, a[8 * sr :], 20.0, 3.0, sr)
    place(mix, a[11 * sr :], 23.0, 2.5, sr)
    place(mix, b[8 * sr :], 23.0, 2.5, sr)
    place(mix, c[4 * sr :], 23.0, 2.5, sr)
    place(mix, c[int(6.5 * sr) :], 25.5, 2.5, sr)
    place(mix, b[int(10.5 * sr) :], 28.0, 2.0, sr)

    mix = mix / (np.abs(mix).max() + 1e-8) * 0.9

    ground_truth = [
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
    return mix, ground_truth


def format_gt(ground_truth):
    lines = [f"{'TIME':>12}  {'SPEAKERS':>12}  DESCRIPTION", "-" * 50]
    for start, end, spks, desc in ground_truth:
        spk_str = ",".join(f"s{s}" for s in spks) if spks else "---"
        lines.append(f"{start:5.1f}-{end:5.1f}s  {spk_str:>12}  {desc}")
    return "\n".join(lines)


os.makedirs("test_audio", exist_ok=True)
speakers = load_speakers(n=3)
mix, gt = build_hard_meeting(speakers)
wav = torch.tensor(mix).unsqueeze(0)
torchaudio.save(config.HARD_AUDIO, wav, config.SAMPLE_RATE)
duration = len(mix) / config.SAMPLE_RATE
print(f"{config.HARD_AUDIO} ({duration:.1f}s)")
print(format_gt(gt))
os._exit(0)
