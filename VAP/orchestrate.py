import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from VAP import dyad
from VAP import router
from VAP import state
from VAP import vap


VAP_EVERY_N = 4
PRINT_EVERY_N = 40
FRAME_SEC = 0.08


def process_file(audio, sample_rate, probs_sequence, vap_model):
    frame_samples = int(FRAME_SEC * sample_rate)
    n_frames = len(probs_sequence)
    n_speakers = len(probs_sequence[0]) if n_frames > 0 else 4

    meeting = state.new_state()
    rtr = router.new_router(n_speakers)
    prev_dyad = None
    frame_log = []
    high_openings = []

    print(f"Processing {n_frames} frames ({n_frames * FRAME_SEC:.1f}s)")

    for i in range(n_frames):
        ts = i * FRAME_SEC
        probs = probs_sequence[i]
        start = i * frame_samples
        end = start + frame_samples
        if end > len(audio):
            frame_audio = np.pad(audio[start:], (0, end - len(audio)))
        else:
            frame_audio = audio[start:end]

        dyad_out = dyad.detect(probs, ts)
        transition = dyad.classify_transition(prev_dyad, dyad_out)
        if transition and transition["type"] == "dyad_shift" and transition["from_pair"]:
            state.increment_pair_turns(meeting, transition["from_pair"])

        router.feed_frame(rtr, frame_audio, probs, dyad_out, ts)
        vap.push_frame(vap_model, frame_audio)
        state.update_speakers(meeting, probs, ts)
        state.update_dyad(meeting, dyad_out)

        if i % VAP_EVERY_N == 0 and i > 0:
            vap_out = vap.get_latest(vap_model, "ai_buffer")
            state.update_vap(meeting, vap_out)

            if vap_out["ai_opening"] is not None and vap_out["ai_opening"] >= config.HIGH_OPENING_THRESHOLD:
                active = [k for k, v in meeting["speakers"].items() if v["is_active"]]
                high_openings.append({
                    "timestamp": ts,
                    "ai_opening": vap_out["ai_opening"],
                    "active_speakers": active,
                    "mode": dyad_out["mode"],
                })

        frame_log.append({
            "frame": i,
            "timestamp": ts,
            "probs": probs,
            "dyad_mode": dyad_out["mode"],
            "active_pair": dyad_out["active_pair"],
            "dominant": dyad_out["dominant"],
            "ai_opening": meeting["vap"]["ai_opening"],
            "turn_hold": meeting["vap"]["turn_hold"],
        })

        if i > 0 and i % PRINT_EVERY_N == 0:
            print(_format_frame_status(meeting, dyad_out))

        prev_dyad = dyad_out

    print(f"\n{'=' * 60}\nFINAL STATE\n{'=' * 60}")
    print(state.render_for_llm(meeting))
    print(_format_router_summary(rtr))
    print(_format_pair_summary(meeting))
    print(f"\nHigh VAP openings: {len(high_openings)} moments above {config.HIGH_OPENING_THRESHOLD}")
    return meeting, frame_log, high_openings


def _format_frame_status(meeting, dyad_out):
    t = meeting["timestamp"]
    mode = dyad_out["mode"]
    dom = dyad_out["dominant"]
    ai = meeting["vap"]["ai_opening"] or 0.5
    hold = meeting["vap"]["turn_hold"] or 0.5
    active = [f"s{k}:{v['current_prob']:.2f}" for k, v in meeting["speakers"].items() if v["is_active"]]
    active_str = ", ".join(active) if active else "silence"
    return f"  [{t:6.2f}s] {mode:7s} dom={dom} active=[{active_str}] vap_open={ai:.2f} hold={hold:.2f}"


def _format_router_summary(rtr):
    lines = ["\nRouter status:"]
    for key, val in router.get_router_status(rtr).items():
        if val["total_frames"] > 0:
            lines.append(f"  {key}: {val['total_active_sec']:.1f}s active, {val['total_frames']} frames")
    return "\n".join(lines)


def _format_pair_summary(meeting):
    lines = ["\nPair history:"]
    for pair, ph in sorted(meeting["pair_history"].items()):
        if ph["total_sec"] > 0:
            lines.append(f"  {pair}: {ph['total_sec']:.1f}s, {ph['turn_count']} turns")
    return "\n".join(lines)
