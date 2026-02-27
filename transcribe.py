import time

import config
from voxtral import load_model as load_voxtral, transcribe_chunk

FRAME_SEC = 0.08


def extract_segments(probs):
    n_frames = len(probs)
    n_spk = len(probs[0]) if n_frames > 0 else 4
    segments = []

    for spk in range(n_spk):
        active = [i for i in range(n_frames) if probs[i][spk] > config.SEGMENT_ACTIVE_THRESHOLD]
        if not active:
            continue

        runs = []
        s, e = active[0], active[0]
        for f in active[1:]:
            if f - e <= config.SEGMENT_MERGE_GAP_FRAMES:
                e = f
            else:
                runs.append((s, e))
                s, e = f, f
        runs.append((s, e))

        for s, e in runs:
            dur = (e - s + 1) * FRAME_SEC
            if dur >= config.SEGMENT_MIN_SEC:
                segments.append({
                    "slot_id": spk,
                    "start_sec": s * FRAME_SEC,
                    "end_sec": (e + 1) * FRAME_SEC,
                })

    segments.sort(key=lambda x: x["start_sec"])
    return segments


def transcribe_all(audio, sr, segments, model=None, proc=None):
    if model is None or proc is None:
        model, proc = load_voxtral()

    results = []
    t0 = time.time()

    for i, seg in enumerate(segments):
        s = int(seg["start_sec"] * sr)
        e = int(seg["end_sec"] * sr)
        text = transcribe_chunk(model, proc, audio[s:e], sr)
        if text:
            results.append({
                "slot_id": seg["slot_id"],
                "speaker": f"slot_{seg['slot_id']}",
                "start": seg["start_sec"],
                "end": seg["end_sec"],
                "text": text,
            })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(segments)} ({time.time() - t0:.0f}s)")

    return results


def get_context(transcript, ts, window=None):
    w = window or config.TRANSCRIPT_WINDOW_SEC
    lo, hi = ts - w, ts + w
    return [t for t in transcript if t["end"] > lo and t["start"] < hi]


def format_context(entries):
    return "\n".join(
        f"  [{e['start']:.1f}-{e['end']:.1f}] {e['speaker']}: {e['text']}"
        for e in entries
    )


def format_openings(openings, transcript):
    lines = []
    for ho in openings:
        ts = ho["timestamp"]
        m, s = int(ts // 60), ts % 60
        reason = ho.get("reason", "threshold")
        active = ", ".join(f"s{x}" for x in ho["active_speakers"])
        lines.append(f"\n[{m}:{s:04.1f}] vap_open={ho['ai_opening']:.2f} "
                     f"mode={ho['mode']} reason={reason} active=[{active}]")
        nearby = get_context(transcript, ts)
        if nearby:
            lines.append(format_context(nearby))
        else:
            lines.append("  (no transcript in window)")
    return "\n".join(lines)
