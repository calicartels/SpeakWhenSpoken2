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
    speaker_audio = {}
    speaker_ids = []

    for item in ds:
        spk = item["speaker_id"]
        audio = np.array(item["audio"]["array"], dtype=np.float32)

        if spk not in speaker_audio:
            if len(speaker_ids) >= n:
                continue
            speaker_audio[spk] = []
            speaker_ids.append(spk)

        if spk in speaker_ids:
            speaker_audio[spk].append(audio)

        if len(speaker_ids) < n:
            continue
        all_long = all(
            sum(len(a) for a in speaker_audio[s]) > 30 * config.SAMPLE_RATE
            for s in speaker_ids
        )
        if all_long:
            break

    speakers = []
    for spk in speaker_ids:
        combined = np.concatenate(speaker_audio[spk])
        speakers.append(combined)
    return speakers


def place(mix, chunk, start_sec, sr):
    s = int(start_sec * sr)
    end = s + len(chunk)
    if end > len(mix):
        chunk = chunk[: len(mix) - s]
        end = s + len(chunk)
    mix[s:end] += chunk


def build_hard_meeting(speakers):
    sr = config.SAMPLE_RATE
    for i, spk in enumerate(speakers):
        assert len(spk) >= 20 * sr, f"spk{i} too short: {len(spk)/sr:.1f}s"

    total_sec = 35
    mix = np.zeros(total_sec * sr, dtype=np.float32)
    a, b, c = speakers[0], speakers[1], speakers[2]
    a_off, b_off, c_off = 0, 0, 0

    def take(src, offset, dur):
        n = int(dur * sr)
        chunk = src[offset : offset + n]
        return chunk, offset + n

    chunk, a_off = take(a, a_off, 3.0)
    place(mix, chunk, 0.0, sr)

    chunk, b_off = take(b, b_off, 3.0)
    place(mix, chunk, 3.0, sr)

    chunk, a_off = take(a, a_off, 2.0)
    place(mix, chunk, 6.0, sr)
    chunk, b_off = take(b, b_off, 2.0)
    place(mix, chunk, 6.0, sr)

    chunk, a_off = take(a, a_off, 1.5)
    place(mix, chunk, 9.0, sr)
    chunk, b_off = take(b, b_off, 1.5)
    place(mix, chunk, 10.5, sr)
    chunk, a_off = take(a, a_off, 1.5)
    place(mix, chunk, 12.0, sr)
    chunk, b_off = take(b, b_off, 1.5)
    place(mix, chunk, 13.5, sr)

    chunk, c_off = take(c, c_off, 4.0)
    place(mix, chunk, 16.0, sr)

    chunk, a_off = take(a, a_off, 3.0)
    place(mix, chunk, 20.0, sr)

    chunk, a_off = take(a, a_off, 2.5)
    place(mix, chunk, 23.0, sr)
    chunk, b_off = take(b, b_off, 2.5)
    place(mix, chunk, 23.0, sr)
    chunk, c_off = take(c, c_off, 2.5)
    place(mix, chunk, 23.0, sr)

    chunk, c_off = take(c, c_off, 2.5)
    place(mix, chunk, 25.5, sr)

    chunk, b_off = take(b, b_off, 2.0)
    place(mix, chunk, 28.0, sr)

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


def save_clean_sources(speakers):
    os.makedirs(config.CLEAN_DIR, exist_ok=True)
    sr = config.SAMPLE_RATE
    clean_segments = [
        ("spk0_early", 0, 0.0, 3.0),
        ("spk0_rapid", 0, 5.0, 6.5),
        ("spk0_return", 0, 8.0, 11.0),
        ("spk1_early", 1, 0.0, 3.0),
        ("spk1_rapid", 1, 5.0, 6.5),
        ("spk1_close", 1, 10.5, 12.5),
        ("spk2_solo", 2, 0.0, 4.0),
        ("spk2_late", 2, 6.5, 9.0),
    ]
    for name, spk_idx, start, end in clean_segments:
        src = speakers[spk_idx]
        s = int(start * sr)
        e = int(end * sr)
        chunk = src[s:e]
        path = os.path.join(config.CLEAN_DIR, f"{name}.wav")
        torchaudio.save(path, torch.tensor(chunk).unsqueeze(0), sr)


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
save_clean_sources(speakers)
duration = len(mix) / config.SAMPLE_RATE
print(f"{config.HARD_AUDIO} ({duration:.1f}s)")
print(format_gt(gt))
os._exit(0)
