import asyncio
import collections
import json
import base64
import logging
import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

import config
import gate
from INPUT_PIPELINE import sortformer, voxtral
from LLM import decide as decide_module
from VAP import vap, dyad, state, router
from LLM import memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("server")

FRAME_SEC = 0.08
FRAME_SAMPLES = int(FRAME_SEC * config.SAMPLE_RATE)

VAP_INTERVAL = 4
GLINER_INTERVAL = 125
GLINER_MIN_WORDS = 10
MAX_AUDIO_BUF_SEC = 60
SEGMENT_GAP_FRAMES = int(1.5 / FRAME_SEC)

models = {}
log_buffer = collections.deque(maxlen=200)
log_clients = set()


def slog(msg):
    log.info(msg)
    entry = {"ts": time.time(), "msg": msg}
    log_buffer.append(entry)


def load_all():
    slog("Loading Sortformer (streaming)...")
    models["sortformer"] = sortformer.load_model()
    slog("Loading VAP (maai)...")
    models["vap"] = vap.load_vap()
    slog(f"Voxtral served by vLLM (port {config.VLLM_PORT})")
    models["voxtral_model"], models["voxtral_proc"] = voxtral.load_model()
    slog("Loading GLiNER...")
    models["gliner"] = memory.load_model()
    slog("All models loaded")


async def handle_client(websocket):
    audio_buf = np.zeros(0, dtype=np.float32)
    frame_count = 0
    meeting = state.new_state()
    rtr = router.new_router(4)
    prev_dyad = None
    mem_store = memory.new_store()
    transcript_accum = []
    cur_seg = {"text": "", "start": 0.0, "end": 0.0, "slot_id": -1, "speaker": "unknown"}
    last_text_frame = 0

    last_probs = [0.0, 0.0, 0.0, 0.0]
    last_gate_time = 0.0
    GATE_COOLDOWN_SEC = 5.0
    vox_stream = await voxtral.new_stream(models["voxtral_model"], models["voxtral_proc"])

    slog("Client connected (vLLM realtime session open)")

    async for message in websocket:
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            continue

        if msg.get("type") == "log_subscribe":
            since = msg.get("since", 0)
            recent = [e for e in log_buffer if e["ts"] > since]
            try:
                await websocket.send(json.dumps({"server_log": recent}))
            except ConnectionClosed:
                break
            continue

        if msg["type"] == "stop":
            break

        if msg["type"] != "audio":
            continue

        raw = base64.b64decode(msg["data"])
        chunk = np.frombuffer(raw, dtype=np.float32)
        audio_buf = np.append(audio_buf, chunk)

        max_audio_samples = MAX_AUDIO_BUF_SEC * config.SAMPLE_RATE
        if len(audio_buf) > max_audio_samples:
            audio_buf = audio_buf[-max_audio_samples:]

        frame_count += 1
        results = {"debug": {"frame": frame_count, "ts": round(frame_count * FRAME_SEC, 2),
                              "chunk_samples": len(chunk), "audio_buf_sec": round(len(audio_buf) / config.SAMPLE_RATE, 1)}}

        if frame_count == 1:
            slog(f"First audio frame received ({len(chunk)} samples)")
        if frame_count % 100 == 0:
            slog(f"Frame {frame_count}, audio_buf={len(audio_buf)/config.SAMPLE_RATE:.1f}s")

        await voxtral.feed_audio(vox_stream, chunk)

        probs_frames = sortformer.push_audio(models["sortformer"], chunk)
        if probs_frames:
            last_probs = probs_frames[-1]
            results["diarization"] = {
                "n_frames": len(probs_frames),
                "latest_probs": last_probs,
            }

        ts = frame_count * FRAME_SEC
        current_probs = last_probs

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
                "speaker_probs": current_probs,
                "silence_gap_sec": meeting["silence"]["current_gap_sec"],
            }

            if gate.should_open(vap_out["ai_opening"], dyad_out["mode"], dom_prob) and ts - last_gate_time >= GATE_COOLDOWN_SEC:
                last_gate_time = ts
                reason = "silence_gap" if dyad_out["mode"] == "silence" else (
                    "speaker_fading" if dom_prob is not None and dom_prob < config.GATE_FADE_PROB else "prosodic_boundary"
                )
                gate_open = {
                    "timestamp": ts,
                    "ai_opening": vap_out["ai_opening"],
                    "mode": dyad_out["mode"],
                    "dominant_prob": dom_prob,
                    "active_speakers": results["state"]["active_speakers"],
                    "reason": reason,
                    "meeting_state": state.render_for_llm(meeting),
                    "memory": memory.render_for_llm(mem_store),
                }
                try:
                    opening = {"timestamp": ts, "ai_opening": vap_out["ai_opening"], "mode": dyad_out["mode"],
                               "active_speakers": results["state"]["active_speakers"], "reason": reason}
                    gate_open["decision"] = await asyncio.to_thread(
                        decide_module.decide,
                        gate_open["meeting_state"], [], gate_open["memory"], opening,
                    )
                except Exception:
                    gate_open["decision"] = {"speak": False, "reason": "LLM call failed", "response": ""}
                results["gate_open"] = gate_open

        if frame_count % GLINER_INTERVAL == 0 and transcript_accum:
            combined = " ".join(t.get("text", "") for t in transcript_accum[-20:])
            if len(combined.split()) >= GLINER_MIN_WORDS:
                entities = await asyncio.to_thread(memory.extract, models["gliner"], combined)
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

        new_text = voxtral.get_text(vox_stream)
        if new_text:
            dom = dyad_out["dominant"] if dyad_out else None
            slot_id = dom if dom is not None else -1
            speaker = f"slot_{dom}" if dom is not None else "unknown"

            if cur_seg["text"] and cur_seg["slot_id"] != slot_id and slot_id != -1:
                transcript_accum.append(dict(cur_seg))
                results.setdefault("transcript", []).append(dict(cur_seg))
                cur_seg = {"text": "", "start": 0.0, "end": 0.0, "slot_id": -1, "speaker": "unknown"}

            if not cur_seg["text"]:
                cur_seg.update(start=round(ts, 2), slot_id=slot_id, speaker=speaker)

            cur_seg["text"] += new_text
            cur_seg["end"] = round(ts, 2)
            last_text_frame = frame_count

            stripped = cur_seg["text"].rstrip()
            if stripped and stripped[-1] in ".?!":
                transcript_accum.append(dict(cur_seg))
                results.setdefault("transcript", []).append(dict(cur_seg))
                cur_seg = {"text": "", "start": 0.0, "end": 0.0, "slot_id": -1, "speaker": "unknown"}

        elif cur_seg["text"] and frame_count - last_text_frame > SEGMENT_GAP_FRAMES:
            transcript_accum.append(dict(cur_seg))
            results.setdefault("transcript", []).append(dict(cur_seg))
            cur_seg = {"text": "", "start": 0.0, "end": 0.0, "slot_id": -1, "speaker": "unknown"}

        try:
            await websocket.send(json.dumps(results, default=_serialize))
        except ConnectionClosed:
            slog("Client disconnected")
            await voxtral.stop_stream(vox_stream)
            break


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
