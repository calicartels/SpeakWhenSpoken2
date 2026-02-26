import json
import time
import resource

import torch
import numpy as np
from nemo.collections.asr.models import SortformerEncLabelModel

import config

# #region agent log
_DBG_LOG = "/tmp/debug-e28f32.log"
def _dbg_sf(msg, hyp, **data):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    gpu_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    gpu_reserved_mb = torch.cuda.memory_reserved() / 1e6 if torch.cuda.is_available() else 0
    entry = {"sessionId": "e28f32", "timestamp": int(time.time() * 1000), "location": "sortformer.py", "message": msg, "hypothesisId": hyp, "data": {**data, "rss_mb": round(rss_mb, 1), "gpu_mb": round(gpu_mb, 1), "gpu_reserved_mb": round(gpu_reserved_mb, 1)}}
    with open(_DBG_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
# #endregion


def load_model():
    model = SortformerEncLabelModel.from_pretrained(config.SORTFORMER_MODEL)
    model.eval()
    model.sortformer_modules.chunk_len = config.SORTFORMER_CHUNK_LEN
    model.sortformer_modules.chunk_right_context = config.SORTFORMER_RIGHT_CONTEXT
    model.sortformer_modules.fifo_len = config.SORTFORMER_FIFO_LEN
    model.sortformer_modules.spkcache_update_period = config.SORTFORMER_CACHE_UPDATE
    model.sortformer_modules.spkcache_len = config.SORTFORMER_CACHE_LEN
    model.sortformer_modules._check_streaming_parameters()
    return model


def diarize(model, audio_path):
    segments, probs = model.diarize(
        audio=[audio_path],
        batch_size=1,
        include_tensor_outputs=True,
    )
    return segments[0], probs[0]


def get_frame_probs(model, audio_path, n_speakers=4):
    # #region agent log
    _dbg_sf("before_diarize", "H2,H3")
    # #endregion
    segments, probs = diarize(model, audio_path)
    # #region agent log
    _dbg_sf("after_diarize", "H2,H3", probs_shape=list(probs.shape), probs_dtype=str(probs.dtype), probs_device=str(probs.device))
    # #endregion
    arr = probs.cpu().numpy()
    # #region agent log
    _dbg_sf("after_probs_cpu_numpy", "H2,H5", arr_shape=list(arr.shape), arr_mb=round(arr.nbytes / 1e6, 2))
    # #endregion
    del probs
    del segments
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out = [list(arr[t]) for t in range(arr.shape[0])]
    # #region agent log
    _dbg_sf("after_list_creation", "H5", n_frames=len(out))
    # #endregion
    del arr
    while len(out[0]) < n_speakers:
        for row in out:
            row.append(0.0)
    return out


def format_segments(segments):
    lines = [f"{'START':>8}  {'END':>8}  SPEAKER", "-" * 30]
    for seg in segments:
        parts = str(seg).split()
        if len(parts) >= 3:
            lines.append(f"{parts[0]:>8}  {parts[1]:>8}  {parts[2]}")
    return "\n".join(lines)


def format_prob_summary(probs):
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    _, T, S = probs.shape
    probs = probs[0]
    duration = T * 0.08
    lines = ["", f"Probability matrix: {T} frames x {S} speakers ({duration:.1f}s)"]
    for s in range(S):
        peak = probs[:, s].max()
        active_frames = (probs[:, s] > 0.5).sum()
        active_sec = active_frames * 0.08
        if peak > 0.1:
            lines.append(f"  spk{s}: peak={peak:.3f}, active={active_sec:.1f}s ({active_frames} frames)")
    lines.append("Sample frames (prob per speaker):")
    indices = np.linspace(0, T - 1, min(8, T), dtype=int)
    spk_hdr = "  ".join(f"{'spk'+str(j):>6}" for j in range(S))
    lines.append(f"  {'time':>6}  {spk_hdr}")
    for i in indices:
        t = i * 0.08
        row = probs[i]
        row_str = "  ".join(f"{row[j]:6.3f}" for j in range(S))
        lines.append(f"  {t:6.2f}  {row_str}")
    return "\n".join(lines)


if __name__ == "__main__":
    model = load_model()
    segments, probs = diarize(model, config.TEST_AUDIO)
    print(format_segments(segments))
    print(format_prob_summary(probs))
