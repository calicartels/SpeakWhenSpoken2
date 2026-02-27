import os
import gc
import time
import argparse

import numpy as np
import torch
import torchaudio

import config
from INPUT_PIPELINE import transcribe
from VAP import orchestrate
from VAP import vap


def load_audio(path):
    wav, sr = torchaudio.load(path)
    if sr != config.SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, config.SAMPLE_RATE)
    wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav[0]
    return wav.numpy().astype(np.float32)


def load_or_compute_probs(audio_path):
    if os.path.exists(config.PROBS_CACHE_PATH):
        arr = np.load(config.PROBS_CACHE_PATH)
        print(f"Loaded cached probs: {arr.shape}")
        return arr.tolist()

    from INPUT_PIPELINE.sortformer import load_model, get_frame_probs
    model = load_model()
    probs = get_frame_probs(model, audio_path)
    np.save(config.PROBS_CACHE_PATH, probs)
    print(f"Cached {len(probs)} frames to {config.PROBS_CACHE_PATH}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return probs


def run(audio_path=None, max_sec=None):
    audio_path = audio_path or config.TEST_AUDIO_20MIN
    audio = load_audio(audio_path)
    print(f"Audio: {len(audio)/config.SAMPLE_RATE:.1f}s")

    if max_sec:
        audio = audio[:int(max_sec * config.SAMPLE_RATE)]

    probs = load_or_compute_probs(audio_path)
    vap_model = vap.load_vap()

    t0 = time.time()
    meeting, _, openings = orchestrate.process_file(
        audio, config.SAMPLE_RATE, probs, vap_model,
    )
    frame_t = time.time() - t0

    segments = transcribe.extract_segments(probs)
    print(f"{len(segments)} segments")

    from INPUT_PIPELINE.voxtral import load_model as load_voxtral
    model, proc = load_voxtral()

    t0 = time.time()
    transcript = transcribe.transcribe_all(audio, config.SAMPLE_RATE, segments, model, proc)
    tx_t = time.time() - t0
    print(f"Transcribed {len(transcript)} segments in {tx_t:.1f}s")

    meeting["transcript"] = transcript

    if openings:
        print(f"\n{'=' * 60}")
        print("TRANSCRIPT CONTEXT AT HIGH VAP OPENINGS")
        print(f"{'=' * 60}")
        print(transcribe.format_openings(openings, transcript))

    print(f"\nTotal: frames {frame_t:.0f}s + transcription {tx_t:.0f}s")


ap = argparse.ArgumentParser()
ap.add_argument("--audio", default=None)
ap.add_argument("--max-sec", type=float, default=None)
args = ap.parse_args()
run(args.audio, args.max_sec)
