
import os
import numpy as np
import torch
import torchaudio
from datasets import load_dataset

import config


def load_librispeech_samples(n=4):
    ds = load_dataset(
        "librispeech_asr", "clean",
        split="test",
        streaming=True,
        trust_remote_code=True,
    )
    samples = []
    seen_speakers = set()
    for item in ds:
        spk = item["speaker_id"]
        if spk not in seen_speakers and len(item["audio"]["array"]) > 3 * config.SAMPLE_RATE:
            samples.append({
                "audio": np.array(item["audio"]["array"], dtype=np.float32),
                "sr": item["audio"]["sampling_rate"],
                "speaker": spk,
                "text": item["text"],
            })
            seen_speakers.add(spk)
        if len(samples) >= n:
            break
    return samples


def mix_speakers(samples):
    sr = config.SAMPLE_RATE
    gap = int(0.5 * sr)
    overlap = int(1.0 * sr)
    segments = []
    timeline = []
    pos = 0
    for i, s in enumerate(samples):
        audio = s["audio"]
        if s["sr"] != sr:
            audio = torchaudio.functional.resample(
                torch.tensor(audio).unsqueeze(0), s["sr"], sr
            ).squeeze(0).numpy()
        audio = audio[:5 * sr]
        start = max(0, pos - overlap) if i > 0 else pos
        segments.append({"audio": audio, "start": start, "speaker": i})
        timeline.append(f"spk{i}: {start/sr:.2f}s - {(start + len(audio))/sr:.2f}s")
        pos = start + len(audio) + gap
    total_len = max(seg["start"] + len(seg["audio"]) for seg in segments)
    mixed = np.zeros(total_len, dtype=np.float32)
    for seg in segments:
        end = seg["start"] + len(seg["audio"])
        mixed[seg["start"]:end] += seg["audio"]
    mixed = mixed / (np.abs(mixed).max() + 1e-8) * 0.9
    return mixed, timeline


def format_timeline(timeline):
    return "\n".join(timeline)


os.makedirs("test_audio", exist_ok=True)
samples = load_librispeech_samples(n=3)
mixed, timeline = mix_speakers(samples)
wav = torch.tensor(mixed).unsqueeze(0)
torchaudio.save(config.TEST_AUDIO, wav, config.SAMPLE_RATE)
duration = len(mixed) / config.SAMPLE_RATE
print(f"{config.TEST_AUDIO} ({duration:.1f}s)")
print(format_timeline(timeline))
