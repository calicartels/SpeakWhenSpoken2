# Full pipeline: Sortformer -> MaAI VAP -> Voxtral transcript
# Usage: python pipeline.py [--audio path] [--max-sec N]

import os
import gc
import time

import numpy as np
import torch
import torchaudio

import config
import transcribe
from VAP import orchestrate
from VAP import vap


def load_audio(path):
    wav, sr = torchaudio.load(path)
    if sr != config.SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, config.SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav[0]
    return wav.numpy().astype(np.float32)


def load_or_compute_probs(audio_path):
    if os.path.exists(config.PROBS_CACHE_PATH):
        arr = np.load(config.PROBS_CACHE_PATH)
        print(f"Loaded cached probs: {arr.shape} from {config.PROBS_CACHE_PATH}")
        return arr.tolist()

    print("Running Sortformer (first time)...")
    from sortformer import load_model, get_frame_probs
    model = load_model()
    probs_list = get_frame_probs(model, audio_path)

    np.save(config.PROBS_CACHE_PATH, probs_list)
    print(f"Cached probs ({len(probs_list)} frames) to {config.PROBS_CACHE_PATH}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return probs_list


def run(audio_path=None, max_sec=None):
    if audio_path is None:
        audio_path = config.TEST_AUDIO_20MIN

    print(f"Loading audio: {audio_path}")
    audio = load_audio(audio_path)
    duration = len(audio) / config.SAMPLE_RATE
    print(f"Audio: {duration:.1f}s, {config.SAMPLE_RATE}Hz")

    if max_sec:
        audio = audio[:int(max_sec * config.SAMPLE_RATE)]
        print(f"Truncated to {max_sec}s")

    probs = load_or_compute_probs(audio_path)

    print("\nLoading MaAI VAP...")
    vap_model = vap.load_vap()

    # Frame loop
    t0 = time.time()
    meeting, frame_log, high_openings = orchestrate.process_file(
        audio, config.SAMPLE_RATE, probs, vap_model,
    )
    frame_sec = time.time() - t0
    print(f"\nFrame loop: {frame_sec:.1f}s")

    # Transcription
    print("\nExtracting segments...")
    segments = transcribe.extract_segments(probs)
    print(f"Found {len(segments)} speaking segments")

    print("Loading Voxtral...")
    from voxtral import load_model as load_voxtral
    voxtral_model, voxtral_proc = load_voxtral()

    t0 = time.time()
    transcript = transcribe.transcribe_all(
        audio, config.SAMPLE_RATE, segments, voxtral_model, voxtral_proc,
    )
    tx_sec = time.time() - t0
    print(f"Transcription: {tx_sec:.1f}s")

    meeting["transcript"] = transcript
    transcribe.print_openings(high_openings, transcript)

    print(f"\nTotal: frame loop {frame_sec:.0f}s + transcription {tx_sec:.0f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=str, default=None)
    ap.add_argument("--max-sec", type=float, default=None)
    args = ap.parse_args()
    run(args.audio, args.max_sec)
