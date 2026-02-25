import torch
import numpy as np
from nemo.collections.asr.models import SortformerEncLabelModel

import config


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


model = load_model()
segments, probs = diarize(model, config.TEST_AUDIO)
print(format_segments(segments))
print(format_prob_summary(probs))
