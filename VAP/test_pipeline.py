import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VAP import orchestrate
from VAP import vap


def mock_probs(duration_sec, scenario="two_speakers"):
    n_frames = int(duration_sec / 0.08)
    probs = []
    if scenario == "two_speakers":
        for i in range(n_frames):
            t = i * 0.08
            cycle = t % 8.0
            if cycle < 3.5:
                probs.append([0.9, 0.05, 0.0, 0.0])
            elif cycle < 4.0:
                probs.append([0.4, 0.5, 0.0, 0.0])
            elif cycle < 7.5:
                probs.append([0.05, 0.9, 0.0, 0.0])
            else:
                probs.append([0.05, 0.1, 0.0, 0.0])
    elif scenario == "four_speakers":
        for i in range(n_frames):
            t = i * 0.08
            phase = t % 20.0
            if phase < 4.0:
                probs.append([0.9, 0.05, 0.02, 0.01])
            elif phase < 4.5:
                probs.append([0.1, 0.1, 0.05, 0.05])
            elif phase < 8.0:
                probs.append([0.1, 0.88, 0.02, 0.01])
            elif phase < 8.3:
                probs.append([0.05, 0.4, 0.7, 0.02])
            elif phase < 11.0:
                probs.append([0.05, 0.05, 0.85, 0.02])
            elif phase < 11.5:
                probs.append([0.05, 0.05, 0.05, 0.05])
            elif phase < 14.0:
                probs.append([0.02, 0.02, 0.05, 0.87])
            elif phase < 14.5:
                probs.append([0.6, 0.02, 0.05, 0.5])
            elif phase < 18.0:
                probs.append([0.88, 0.05, 0.02, 0.02])
            else:
                probs.append([0.1, 0.08, 0.12, 0.05])
    elif scenario == "silence_test":
        for i in range(n_frames):
            t = i * 0.08
            if t < 5.0:
                probs.append([0.9, 0.05, 0.0, 0.0])
            elif t < 10.0:
                probs.append([0.02, 0.02, 0.0, 0.0])
            elif t < 15.0:
                probs.append([0.05, 0.88, 0.0, 0.0])
            else:
                probs.append([0.05, 0.05, 0.0, 0.0])
    return probs


def mock_audio(probs, sample_rate=16000):
    frame_samples = int(0.08 * sample_rate)
    audio = np.zeros(len(probs) * frame_samples, dtype=np.float32)
    for i, p in enumerate(probs):
        energy = max(p)
        start = i * frame_samples
        end = start + frame_samples
        audio[start:end] = np.random.randn(frame_samples).astype(np.float32) * energy * 0.3
    return audio


def run_mock(duration_sec=20.0, scenario="four_speakers"):
    print(f"Mock: {scenario}, {duration_sec}s\n{'=' * 60}")
    probs = mock_probs(duration_sec, scenario)
    audio = mock_audio(probs)
    vap_model = vap.load_vap()
    meeting, log = orchestrate.process_file(audio, 16000, probs, vap_model)
    modes = {}
    for e in log:
        m = e["dyad_mode"]
        modes[m] = modes.get(m, 0) + 1
    total = len(log)
    print("\nDyad mode distribution:")
    for m, count in sorted(modes.items()):
        print(f"  {m}: {count}/{total} ({100 * count / total:.1f}%)")
    openings = [e for e in log if e["ai_opening"] > 0.6]
    if openings:
        print("\nAI opening moments (ai_opening > 0.6):")
        for o in openings[:10]:
            print(f"  [{o['timestamp']:.2f}s] opening={o['ai_opening']:.2f} mode={o['dyad_mode']}")
    else:
        print("\nNo strong AI openings")
    return meeting, log


def run_real(audio_path=None):
    import torch
    import torchaudio

    import config
    from sortformer import load_model, get_frame_probs

    if audio_path is None:
        audio_path = config.HARD_AUDIO
    if not os.path.exists(audio_path):
        print(f"Audio not found: {audio_path}")
        return None, None

    wav, sr = torchaudio.load(audio_path)
    wav = wav.squeeze(0)
    if wav.dim() == 0:
        wav = wav.unsqueeze(0)
    if sr != config.SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, config.SAMPLE_RATE)
        sr = config.SAMPLE_RATE
    audio = wav.numpy().astype(np.float32)

    model = load_model()
    probs_list = get_frame_probs(model, audio_path)
    vap_model = vap.load_vap()
    return orchestrate.process_file(audio, sr, probs_list, vap_model)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="Use real Sortformer on config.HARD_AUDIO")
    ap.add_argument("--audio", type=str, default=None, help="Audio path for --real")
    args = ap.parse_args()

    if args.real:
        run_real(args.audio)
    else:
        run_mock(20.0, "four_speakers")
        print("\n\n")
        run_mock(20.0, "silence_test")
