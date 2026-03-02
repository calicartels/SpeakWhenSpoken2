import numpy as np
from itertools import combinations


BUFFER_SEC = 8.0
SAMPLE_RATE = 16000
BUFFER_SAMPLES = int(BUFFER_SEC * SAMPLE_RATE)
FRAME_SAMPLES = int(0.08 * SAMPLE_RATE)


def new_ring():
    return {
        "ch1": np.zeros(BUFFER_SAMPLES, dtype=np.float32),
        "ch2": np.zeros(BUFFER_SAMPLES, dtype=np.float32),
        "write_pos": 0,
        "total_frames": 0,
        "last_active": None,
        "total_active_sec": 0.0,
    }


def new_router(n_speakers=4):
    pairs = list(combinations(range(n_speakers), 2))
    return {
        "pair_buffers": {pair: new_ring() for pair in pairs},
        "ai_buffer": new_ring(),
        "n_speakers": n_speakers,
        "last_fed": None,
    }


def append_ring(ring, ch1_frame, ch2_frame):
    n = len(ch1_frame)
    pos = ring["write_pos"]
    buf_len = len(ring["ch1"])
    if pos + n <= buf_len:
        ring["ch1"][pos:pos + n] = ch1_frame
        ring["ch2"][pos:pos + n] = ch2_frame
    else:
        first = buf_len - pos
        ring["ch1"][pos:] = ch1_frame[:first]
        ring["ch1"][:n - first] = ch1_frame[first:]
        ring["ch2"][pos:] = ch2_frame[:first]
        ring["ch2"][:n - first] = ch2_frame[first:]
    ring["write_pos"] = (pos + n) % buf_len
    ring["total_frames"] += 1


def read_ring(ring):
    pos = ring["write_pos"]
    return np.roll(ring["ch1"], -pos), np.roll(ring["ch2"], -pos)


def feed_frame(router, audio_frame, probs, dyad_out, timestamp):
    n = min(len(audio_frame), FRAME_SAMPLES)
    frame = audio_frame[:n]
    ai = router["ai_buffer"]
    append_ring(ai, frame, np.zeros(n, dtype=np.float32))
    ai["last_active"] = timestamp
    ai["total_active_sec"] += 0.08

    fed_pair = None
    if dyad_out["mode"] == "dyad":
        pair = dyad_out["active_pair"]
        if pair in router["pair_buffers"]:
            ring = router["pair_buffers"][pair]
            ch1 = frame * probs[pair[0]]
            ch2 = frame * probs[pair[1]]
            append_ring(ring, ch1, ch2)
            ring["last_active"] = timestamp
            ring["total_active_sec"] += 0.08
            fed_pair = pair
    elif dyad_out["mode"] == "solo":
        dom = dyad_out["dominant"]
        if dom is not None:
            for pair, ring in router["pair_buffers"].items():
                if dom in pair and ring["total_frames"] > 0:
                    if dom == pair[0]:
                        ch1, ch2 = frame * probs[dom], np.zeros(n, dtype=np.float32)
                    else:
                        ch1, ch2 = np.zeros(n, dtype=np.float32), frame * probs[dom]
                    append_ring(ring, ch1, ch2)
                    ring["last_active"] = timestamp

    router["last_fed"] = fed_pair
    return fed_pair


def get_ai_audio(router):
    return read_ring(router["ai_buffer"])


def get_router_status(router):
    status = {}
    for pair, ring in router["pair_buffers"].items():
        status[pair] = {"total_frames": ring["total_frames"], "total_active_sec": ring["total_active_sec"], "last_active": ring["last_active"]}
    status["ai"] = {"total_frames": router["ai_buffer"]["total_frames"], "total_active_sec": router["ai_buffer"]["total_active_sec"]}
    return status
