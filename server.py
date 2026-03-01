import asyncio
import collections
import json
import base64
import logging
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
from core import gate, graph, graph_persist, decide
from INPUT_PIPELINE import sortformer, voxtral
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


def slog(msg):
    log.info(msg)
    log_buffer.append({"ts": time.time(), "msg": msg})


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


# --- Draft helpers ---

def _new_draft():
    return {"text": None, "timestamp": 0.0, "generating": False, "task": None}


def _release_draft(d):
    if d["text"] is None:
        return None
    if time.time() - d["timestamp"] > config.DRAFT_STALE_SEC:
        d["text"] = None
        return None
    text = d["text"]
    d["text"] = None
    return None if text == "SILENT" else text


async def _start_draft(d, transcript, mem_text, state_text, cross_text):
    if d["generating"]:
        return

    async def _gen():
        d["generating"] = True
        d["text"] = await asyncio.to_thread(
            decide.draft, transcript, mem_text, state_text, cross_text)
        d["timestamp"] = time.time()
        d["generating"] = False

    d["task"] = asyncio.create_task(_gen())


def _check_wake(transcript):
    if not transcript:
        return False
    words = []
    for seg in reversed(transcript):
        words = seg.get("text", "").lower().split() + words
        if len(words) >= config.WAKE_WORD_WINDOW:
            break
    tail = " ".join(words[-config.WAKE_WORD_WINDOW:])
    return any(p in tail for p in config.WAKE_PHRASES)


def _fetch_cross_meeting(transcript, kg):
    if not config.SUPERMEMORY_ENABLED:
        return ""
    labels = [n["label"] for n in sorted(
        kg["nodes"].values(), key=lambda n: n["last_seen"], reverse=True)[:5]]
    if transcript:
        labels.append(transcript[-1].get("text", ""))
    query = " ".join(labels).strip()
    if not query:
        return ""
    facts = graph_persist.search_past(query, limit=config.CROSS_MEETING_SEARCH_LIMIT)
    return graph_persist.render_cross_meeting(facts)


# --- Main handler ---

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
    draft_st = _new_draft()
    last_gate_time = 0.0
    mem_text = ""
    state_text = ""
    cross_text = ""

    kg = graph.new_graph()
    participants = set()

    vox_stream = await voxtral.new_stream(models["voxtral_model"], models["voxtral_proc"])
    if vox_stream is None:
        log.error(f"Cannot accept client: vLLM stream not created for meeting {kg['meeting_id']}")
        await websocket.close(1011, "vLLM unavailable")
        return
    slog(f"Client connected, meeting={kg['meeting_id']}")

    try:
        async for message in websocket:
            try:
                msg = json.loads(message)
            except json.JSONDecodeError as e:
                log.error(f"Failed to decode message: {e}")
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
            if len(audio_buf) > MAX_AUDIO_BUF_SEC * config.SAMPLE_RATE:
                audio_buf = audio_buf[-MAX_AUDIO_BUF_SEC * config.SAMPLE_RATE:]
            log.debug(f"Received audio chunk: {len(chunk)} samples")

            frame_count += 1
            ts = frame_count * FRAME_SEC
            results = {"debug": {"frame": frame_count, "ts": round(ts, 2)}}

            if frame_count == 1:
                slog(f"First audio frame ({len(chunk)} samples)")
            if frame_count % 100 == 0:
                slog(f"Frame {frame_count}, buf={len(audio_buf)/config.SAMPLE_RATE:.1f}s")

            # --- Streaming Sortformer ---
            if vox_stream:
                await voxtral.feed_audio(vox_stream, chunk)
            probs_frames = sortformer.push_audio(models["sortformer"], chunk)
            if probs_frames:
                last_probs = probs_frames[-1]
                results["diarization"] = {"n_frames": len(probs_frames), "latest_probs": last_probs}
                log.debug(f"Sortformer returned {len(probs_frames)} frames. Latest probs: {last_probs}")

            current_probs = last_probs

            # --- VAP ---
            frame_start = max(0, len(audio_buf) - FRAME_SAMPLES)
            frame_audio = audio_buf[frame_start:frame_start + FRAME_SAMPLES]
            if len(frame_audio) < FRAME_SAMPLES:
                frame_audio = np.pad(frame_audio, (0, FRAME_SAMPLES - len(frame_audio)))
            vap.push_frame(models["vap"], frame_audio)

            # --- State ---
            dyad_out = dyad.detect(current_probs, ts)
            transition = dyad.classify_transition(prev_dyad, dyad_out)
            if transition and transition["type"] == "dyad_shift" and transition.get("from_pair"):
                state.increment_pair_turns(meeting, transition["from_pair"])
            router.feed_frame(rtr, frame_audio, current_probs, dyad_out, ts)
            state.update_speakers(meeting, current_probs, ts)
            state.update_dyad(meeting, dyad_out)
            for k, v in meeting["speakers"].items():
                if v["is_active"]:
                    participants.add(k)

            # --- VAP tick (every 4 frames) ---
            if frame_count % VAP_INTERVAL == 0 and frame_count > 0:
                vap_out = vap.get_latest(models["vap"], "ai_buffer")
                state.update_vap(meeting, vap_out)
                dom = dyad_out["dominant"]
                dom_prob = meeting["speakers"][dom]["current_prob"] if dom is not None and dom in meeting["speakers"] else None
                silence_sec = meeting["silence"]["current_gap_sec"]

                results["vap"] = {
                    "ai_opening": vap_out["ai_opening"],
                    "turn_hold": vap_out["turn_hold"],
                    "confidence": vap_out["confidence"],
                    "timestamp_ms": int(ts * 1000),
                }
                results["state"] = {
                    "mode": dyad_out["mode"], "dominant": dom,
                    "dominant_prob": dom_prob,
                    "active_speakers": [k for k, v in meeting["speakers"].items() if v["is_active"]],
                    "speaker_probs": current_probs,
                    "silence_gap_sec": silence_sec,
                }
                
                # Expose live draft and generation state to the frontend
                results["live_draft"] = {
                    "text": draft_st["text"],
                    "generating": draft_st["generating"]
                }
                
                state_text = state.render_for_llm(meeting)

                # Pre-warm Mercury 2 draft
                if (vap_out["ai_opening"] >= config.PREWARM_THRESHOLD
                        and not draft_st["generating"] and draft_st["text"] is None):
                    log.info(f"Pre-warming Mercury draft (VAP opening {vap_out['ai_opening']} >= {config.PREWARM_THRESHOLD})")
                    await _start_draft(draft_st, transcript_accum, mem_text, state_text, cross_text)

                # Wake word path
                if _check_wake(transcript_accum):
                    log.info("Wake word detected!")
                    draft_text = _release_draft(draft_st)
                    if draft_text:
                        log.info(f"Using pre-warmed draft for wake word: {draft_text}")
                        results["wake_response"] = {"text": draft_text, "source": "draft", "timestamp": ts}
                    else:
                        log.info("No draft ready, calling Mercury directly for wake word response")
                        resp = await asyncio.to_thread(
                            decide.respond_direct, transcript_accum, mem_text, state_text, cross_text)
                        results["wake_response"] = {"text": resp, "source": "direct", "timestamp": ts}

                # Gate path (simplified)
                elif gate.should_open(vap_out["ai_opening"], dyad_out["mode"],
                                      dom_prob, silence_sec, vap_out["turn_hold"]):
                    now = time.time()
                    if now - last_gate_time >= config.GATE_COOLDOWN_SEC:
                        log.info(f"Gate opened via signal! VAP={vap_out['ai_opening']}, Silence={silence_sec}s")
                        last_gate_time = now
                        draft_text = None
                        if draft_st["generating"] and draft_st["task"]:
                            log.info("Gate opened but draft is still generating. Awaiting HTTP response...")
                            await draft_st["task"]
                            draft_text = _release_draft(draft_st)
                        else:
                            draft_text = _release_draft(draft_st)
                            if not draft_text:
                                log.info("No pre-warmed draft available. Fetching JIT response from Mercury...")
                                draft_text = await asyncio.to_thread(
                                    decide.draft, transcript_accum, mem_text, state_text, cross_text)
                                if draft_text == "SILENT":
                                    draft_text = None

                        log.info(f"Gate response: {draft_text or 'SILENT / stale'}")
                        if draft_text:
                            results["gate_open"] = {
                                "timestamp": ts,
                                "ai_opening": vap_out["ai_opening"],
                                "mode": dyad_out["mode"],
                                "active_speakers": results["state"]["active_speakers"],
                                "decision": draft_text,
                                "source": "draft",
                            }
                    else:
                        log.debug("Gate signal fired but in cooldown.")

            # --- GLiNER + graph (every ~10s) ---
            if frame_count % GLINER_INTERVAL == 0 and transcript_accum:
                combined = " ".join(t.get("text", "") for t in transcript_accum[-20:])
                if len(combined.split()) >= GLINER_MIN_WORDS:
                    ents = await asyncio.to_thread(memory.extract, models["gliner"], combined)
                    if ents:
                        for ent in ents:
                            memory.update(mem_store, {"speaker": "live", "start": ts, "end": ts}, [ent])
                        graph.ingest_gliner(kg, ents, ts)
                        results["entities"] = [{"text": e["text"], "label": e["label"], "score": e["score"]} for e in ents]

                        if config.SUPERMEMORY_ENABLED:
                            await asyncio.to_thread(graph_persist.persist_graph, kg, list(participants))
                            cross_text = await asyncio.to_thread(_fetch_cross_meeting, transcript_accum, kg)

                        mem_text = memory.render_for_llm(mem_store)
                        results["memory"] = mem_text
                        results["graph_text"] = graph.render_for_llm(kg)

            prev_dyad = dyad_out

            # --- Voxtral transcript accumulation ---
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
                break

    finally:
        await voxtral.stop_stream(vox_stream)
        if draft_st["task"] and not draft_st["task"].done():
            draft_st["task"].cancel()
        if config.SUPERMEMORY_ENABLED and kg["nodes"]:
            slog(f"Persisting graph: {len(kg['nodes'])} nodes, {len(kg['edges'])} edges")
            graph_persist.persist_graph(kg, list(participants))
        slog(f"Meeting ended: {kg['meeting_id']}")


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
    async with websockets.serve(handle_client, "0.0.0.0", config.WS_PORT):
        print(f"Server listening on ws://0.0.0.0:{config.WS_PORT}")
        await asyncio.Future()


asyncio.run(main())
