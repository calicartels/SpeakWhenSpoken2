import asyncio
import json
import base64
import sys
import os

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHUNK_MS = 80
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000

GATE_COOLDOWN_SEC = 5.0


async def stream(server_url):
    import websockets

    audio_queue = asyncio.Queue()
    last_gate_time = 0.0

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Audio: {status}")
        audio_queue.put_nowait(indata[:, 0].copy())

    mic = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
        callback=audio_callback,
    )

    async with websockets.connect(server_url) as ws:
        mic.start()
        print(f"Streaming to {server_url}")
        print(f"Mic: {sd.query_devices(sd.default.device[0])['name']}")
        print("Ctrl+C to stop\n")

        send_task = asyncio.create_task(_send_loop(ws, audio_queue))
        recv_task = asyncio.create_task(_recv_loop(ws))

        try:
            await asyncio.gather(send_task, recv_task)
        except asyncio.CancelledError:
            pass
        finally:
            mic.stop()
            await ws.send(json.dumps({"type": "stop"}))


async def _send_loop(ws, audio_queue):
    while True:
        chunk = await audio_queue.get()
        encoded = base64.b64encode(chunk.astype(np.float32).tobytes()).decode("ascii")
        await ws.send(json.dumps({"type": "audio", "data": encoded}))


async def _recv_loop(ws):
    last_gate_time = 0.0

    while True:
        message = await ws.recv()
        results = json.loads(message)

        if "vap" in results:
            v = results["vap"]
            ts_sec = v["timestamp_ms"] / 1000
            opening = v["ai_opening"] or 0.0
            hold = v["turn_hold"] or 0.0

            st = results.get("state", {})
            mode = st.get("mode", "?")
            active = st.get("active_speakers", [])
            silence = st.get("silence_gap_sec", 0)

            if opening > 0.7 or silence > 2.0:
                active_str = ",".join(str(s) for s in active) if active else "none"
                print(f"  [{ts_sec:6.1f}s] {mode:7s} "
                      f"open={opening:.2f} hold={hold:.2f} "
                      f"active=[{active_str}] silence={silence:.1f}s")

        if "gate_open" in results:
            g = results["gate_open"]
            ts = g["timestamp"]

            if ts - last_gate_time < GATE_COOLDOWN_SEC:
                continue
            last_gate_time = ts

            m, s = int(ts // 60), ts % 60
            print(f"\n  GATE OPEN [{m}:{s:04.1f}] "
                  f"vap={g['ai_opening']:.2f} mode={g['mode']}")

            decision = _call_llm(g)
            if decision:
                speak = decision.get("speak", False)
                tag = "SPEAK" if speak else "SILENT"
                print(f"  -> {tag}: {decision.get('reason', '')}")
                if speak:
                    response = decision.get("response", "")
                    print(f"  -> \"{response}\"")

        if "entities" in results:
            ents = results["entities"]
            if ents:
                labels = [f"{e['label']}:{e['text']}" for e in ents[:5]]
                print(f"  Memory: {', '.join(labels)}")


def _call_llm(gate_event):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {"speak": False, "reason": "no API key"}

    sys.path.insert(0, os.path.dirname(__file__))
    from LLM.decide import decide

    state_str = gate_event.get("meeting_state", "MEETING STATE: live session")
    memory_str = gate_event.get("memory", "")

    opening = {
        "timestamp": gate_event["timestamp"],
        "ai_opening": gate_event["ai_opening"],
        "mode": gate_event["mode"],
        "active_speakers": gate_event.get("active_speakers", []),
        "reason": gate_event.get("reason", "live"),
    }

    return decide(state_str, [], memory_str, opening)


host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
port = sys.argv[2] if len(sys.argv) > 2 else "8765"
url = f"ws://{host}:{port}"

asyncio.run(stream(url))
