import os
import sys
import json
import time
import resource

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VAP import orchestrate
from VAP import vap

# #region agent log
_DBG_LOG = "/Users/vishnumukundan/Documents/Duke Code/SS2/.cursor/debug-e28f32.log"
def _dbg(msg, hyp, **data):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    try:
        import torch
        gpu_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        gpu_reserved_mb = torch.cuda.memory_reserved() / 1e6 if torch.cuda.is_available() else 0
    except Exception:
        gpu_mb, gpu_reserved_mb = 0, 0
    entry = {"sessionId": "e28f32", "timestamp": int(time.time() * 1000), "location": "test_pipeline.py", "message": msg, "hypothesisId": hyp, "data": {**data, "rss_mb": round(rss_mb, 1), "gpu_mb": round(gpu_mb, 1), "gpu_reserved_mb": round(gpu_reserved_mb, 1)}}
    with open(_DBG_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
# #endregion


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


def run_real(audio_path=None, max_sec=None):
    import config

    # #region agent log
    _dbg("run_real_start", "H1")
    # #endregion

    if audio_path is None:
        audio_path = getattr(config, "TEST_AUDIO_ASSET", None) or config.HARD_AUDIO
    if not os.path.exists(audio_path):
        print(f"Audio not found: {audio_path}")
        return None, None

    try:
        import torchaudio
        wav, sr = torchaudio.load(audio_path)
        wav = wav.squeeze(0)
        if wav.dim() == 0:
            wav = wav.unsqueeze(0)
        if sr != config.SAMPLE_RATE:
            import torch
            wav = torchaudio.functional.resample(wav, sr, config.SAMPLE_RATE)
            sr = config.SAMPLE_RATE
        audio = wav.numpy().astype(np.float32)
    except Exception as e:
        import soundfile as sf
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != config.SAMPLE_RATE:
            from scipy.signal import resample
            n = int(len(data) * config.SAMPLE_RATE / sr)
            data = resample(data, n)
            sr = config.SAMPLE_RATE
        audio = data.astype(np.float32)

    # #region agent log
    _dbg("audio_loaded", "H1", audio_len=len(audio), audio_mb=round(audio.nbytes / 1e6, 1))
    # #endregion

    if max_sec:
        n_samples = int(max_sec * config.SAMPLE_RATE)
        audio = audio[:n_samples]
        import tempfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, config.SAMPLE_RATE)
            sortformer_path = f.name
        try:
            from sortformer import load_model, get_frame_probs
            model = load_model()
            probs_list = get_frame_probs(model, sortformer_path)
        finally:
            os.unlink(sortformer_path)
        print(f"Truncated to first {max_sec}s")
    else:
        from sortformer import load_model, get_frame_probs
        # #region agent log
        _dbg("before_model_load", "H1")
        # #endregion
        model = load_model()
        # #region agent log
        _dbg("after_model_load", "H1")
        # #endregion
        probs_list = get_frame_probs(model, audio_path)
        # #region agent log
        _dbg("after_get_frame_probs", "H1,H2,H3,H5", n_frames=len(probs_list), probs_sample=probs_list[0] if probs_list else None)
        # #endregion

    # #region agent log
    _dbg("before_del_model", "H1,H3", model_type=type(model).__name__)
    # #endregion

    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    # #region agent log
    _dbg("after_del_model", "H1,H3")
    # #endregion

    sr = config.SAMPLE_RATE
    vap_model = vap.load_vap()

    # #region agent log
    _dbg("before_orchestrate", "H1,H4")
    # #endregion

    result = orchestrate.process_file(audio, sr, probs_list, vap_model)

    # #region agent log
    _dbg("after_orchestrate", "H4")
    # #endregion

    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="Use real Sortformer")
    ap.add_argument("--audio", type=str, default=None, help="Audio path for --real")
    ap.add_argument("--max-sec", type=float, default=None, help="Process only first N seconds")
    args = ap.parse_args()

    if args.real:
        run_real(args.audio, args.max_sec)
    else:
        run_mock(20.0, "four_speakers")
        print("\n\n")
        run_mock(20.0, "silence_test")
