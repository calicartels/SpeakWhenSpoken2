import time

import numpy as np

import config
from voxtral import load_model as load_voxtral, transcribe_chunk


ACTIVE_THRESHOLD = 0.3
MERGE_GAP_FRAMES = 5   # 0.4s at 80ms per frame
MIN_SEGMENT_SEC = 0.5
FRAME_SEC = 0.08


def extract_segments(sortformer_probs):
    n_frames = len(sortformer_probs)
    n_speakers = len(sortformer_probs[0]) if n_frames > 0 else 4
    segments = []

    for spk in range(n_speakers):
        active = [i for i in range(n_frames) if sortformer_probs[i][spk] > ACTIVE_THRESHOLD]
        if not active:
            continue

        runs = []
        start, end = active[0], active[0]
        for f in active[1:]:
            if f - end <= MERGE_GAP_FRAMES:
                end = f
            else:
                runs.append((start, end))
                start, end = f, f
        runs.append((start, end))

        for s, e in runs:
            dur = (e - s + 1) * FRAME_SEC
            if dur >= MIN_SEGMENT_SEC:
                segments.append({
                    "slot_id": spk,
                    "start_sec": s * FRAME_SEC,
                    "end_sec": (e + 1) * FRAME_SEC,
                })

    segments.sort(key=lambda s: s["start_sec"])
    return segments


def transcribe_all(audio, sr, segments, model=None, processor=None):
    if model is None or processor is None:
        model, processor = load_voxtral()

    results = []
    t0 = time.time()

    for i, seg in enumerate(segments):
        s = int(seg["start_sec"] * sr)
        e = int(seg["end_sec"] * sr)
        chunk = audio[s:e]
        text = transcribe_chunk(model, processor, chunk, sr)
        if text:
            results.append({
                "slot_id": seg["slot_id"],
                "speaker": f"slot_{seg['slot_id']}",
                "start": seg["start_sec"],
                "end": seg["end_sec"],
                "text": text,
            })
        if (i + 1) % 20 == 0:
            print(f"  Transcribed {i+1}/{len(segments)} ({time.time() - t0:.0f}s)")

    print(f"Transcribed {len(results)} segments in {time.time() - t0:.1f}s")
    return results


def get_context(transcript, timestamp, window_sec=None):
    if window_sec is None:
        window_sec = config.TRANSCRIPT_WINDOW_SEC
    lo = timestamp - window_sec
    hi = timestamp + window_sec
    return [t for t in transcript if t["end"] > lo and t["start"] < hi]


def format_context(entries):
    return "\n".join(
        f"  [{e['start']:.1f}-{e['end']:.1f}] {e['speaker']}: {e['text']}"
        for e in entries
    )


def print_openings(high_openings, transcript):
    if not high_openings:
        print("No high VAP openings found")
        return

    print(f"\n{'=' * 60}")
    print("TRANSCRIPT CONTEXT AT HIGH VAP OPENINGS")
    print(f"{'=' * 60}")

    for ho in high_openings:
        ts = ho["timestamp"]
        mins, secs = int(ts // 60), ts % 60
        active_str = ", ".join(f"s{s}" for s in ho["active_speakers"])
        nearby = get_context(transcript, ts)

        print(f"\n[{mins}:{secs:04.1f}] vap_open={ho['ai_opening']:.2f} "
              f"mode={ho['mode']} active=[{active_str}]")
        if nearby:
            print(format_context(nearby))
        else:
            print("  (no transcript in window)")
