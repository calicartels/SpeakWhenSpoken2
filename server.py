import asyncio
import json
import base64
import tempfile
import os

import numpy as np
import soundfile as sf
import websockets

import config
import gate
from INPUT_PIPELINE.sortformer import load_model as load_sortformer, diarize
from VAP import vap, dyad, state, router
from LLM import memory

FRAME_SEC = 0.08
FRAME_SAMPLES = int(FRAME_SEC * config.SAMPLE_RATE)

# Run Sortformer every 5s (62 frames). Needs longer segments for accuracy.
DIAR_INTERVAL_FRAMES = 62

# Run VAP every 4 frames (320ms) — same as orchestrate.py.
VAP_INTERVAL = 4

# Run GLiNER every 125 frames (~10s) on accumulated transcript.
GLINER_INTERVAL = 125

models = {}


def load_all():
    print("Loading Sortformer...")
    models["sortformer"] = load_sortformer()
    print("Loading VAP (maai)...")
    models["vap"] = vap.load_vap()
    print("Loading GLiNER...")
    models["gliner"] = memory.load_model()
    print("All models loaded")


async def handle_client(websocket):
    audio_buf = np.zeros(0, dtype=np.float32)
    frame_count = 0
    meeting = state.new_state()
    rtr = router.new_router(4)
    prev_dyad = None
    mem_store = memory.new_store()
    transcript_accum = []

    # Sortformer needs a file path — accumulate audio and write periodically.
    diar_buf = np.zeros(0, dtype=np.float32)
    last_probs = [[0.0, 0.0, 0.0, 0.0]]

    async for message in websocket:
        msg = json.loads(message)

        if msg["type"] == "stop":
            break

        if msg["type"] != "audio":
            continue

        chunk = np.frombuffer(base64.b64decode(msg["data"]), dtype=np.float32)
        audio_buf = np.append(audio_buf, chunk)
        diar_buf = np.append(diar_buf, chunk)

        max_diar_samples = 30 * config.SAMPLE_RATE
        if len(diar_buf) > max_diar_samples:
            diar_buf = diar_buf[-max_diar_samples:]

        frame_count += 1
        results = {}

        if frame_count % DIAR_INTERVAL_FRAMES == 0 and len(diar_buf) > config.SAMPLE_RATE * 2:
            probs_list = _run_sortformer(diar_buf)
            if probs_list:
                last_probs = probs_list
                results["diarization"] = {
                    "n_frames": len(probs_list),
                    "latest_probs": probs_list[-1],
                }

        ts = frame_count * FRAME_SEC
        probs_idx = min(frame_count - 1, len(last_probs) - 1)
        current_probs = last_probs[probs_idx] if probs_idx >= 0 else [0.0, 0.0, 0.0, 0.0]

        frame_start = max(0, len(audio_buf) - FRAME_SAMPLES)
        frame_audio = audio_buf[frame_start:frame_start + FRAME_SAMPLES]
        if len(frame_audio) < FRAME_SAMPLES:
            frame_audio = np.pad(frame_audio, (0, FRAME_SAMPLES - len(frame_audio)))

        vap.push_frame(models["vap"], frame_audio)

        dyad_out = dyad.detect(current_probs, ts)
        transition = dyad.classify_transition(prev_dyad, dyad_out)
        if transition and transition["type"] == "dyad_shift" and transition.get("from_pair"):
            state.increment_pair_turns(meeting, transition["from_pair"])

        router.feed_frame(rtr, frame_audio, current_probs, dyad_out, ts)
        state.update_speakers(meeting, current_probs, ts)
        state.update_dyad(meeting, dyad_out)

        if frame_count % VAP_INTERVAL == 0 and frame_count > 0:
            vap_out = vap.get_latest(models["vap"], "ai_buffer")
            state.update_vap(meeting, vap_out)

            dom = dyad_out["dominant"]
            dom_prob = None
            if dom is not None and dom in meeting["speakers"]:
                dom_prob = meeting["speakers"][dom]["current_prob"]

            results["vap"] = {
                "ai_opening": vap_out["ai_opening"],
                "turn_hold": vap_out["turn_hold"],
                "confidence": vap_out["confidence"],
                "timestamp_ms": int(ts * 1000),
            }

            results["state"] = {
                "mode": dyad_out["mode"],
                "dominant": dyad_out["dominant"],
                "dominant_prob": dom_prob,
                "active_speakers": [
                    k for k, v in meeting["speakers"].items() if v["is_active"]
                ],
                "silence_gap_sec": meeting["silence"]["current_gap_sec"],
            }

            if gate.should_open(vap_out["ai_opening"], dyad_out["mode"], dom_prob):
                reason = "silence_gap" if dyad_out["mode"] == "silence" else (
                    "speaker_fading" if dom_prob is not None and dom_prob < config.GATE_FADE_PROB else "prosodic_boundary"
                )
                results["gate_open"] = {
                    "timestamp": ts,
                    "ai_opening": vap_out["ai_opening"],
                    "mode": dyad_out["mode"],
                    "dominant_prob": dom_prob,
                    "active_speakers": results["state"]["active_speakers"],
                    "reason": reason,
                    "meeting_state": state.render_for_llm(meeting),
                    "memory": memory.render_for_llm(mem_store),
                }

        if frame_count % GLINER_INTERVAL == 0 and transcript_accum:
            combined = " ".join(t.get("text", "") for t in transcript_accum[-20:])
            entities = memory.extract(models["gliner"], combined)
            if entities:
                for ent in entities:
                    seg_stub = {"speaker": "live", "start": ts, "end": ts}
                    memory.update(mem_store, seg_stub, [ent])
                results["entities"] = [
                    {"text": e["text"], "label": e["label"], "score": e["score"]}
                    for e in entities
                ]
                results["memory"] = memory.render_for_llm(mem_store)

        prev_dyad = dyad_out

        if results:
            await websocket.send(json.dumps(results, default=_serialize))


def _run_sortformer(audio_np):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        sf.write(tmp.name, audio_np, config.SAMPLE_RATE)
        segments, probs = diarize(models["sortformer"], tmp.name)
        arr = probs.cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        return [list(arr[t]) for t in range(arr.shape[0])]
    finally:
        os.unlink(tmp.name)


def _serialize(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


async def main():
    load_all()
    async with websockets.serve(handle_client, "0.0.0.0", 8765):
        print("Server listening on ws://0.0.0.0:8765")
        await asyncio.Future()


asyncio.run(main())
