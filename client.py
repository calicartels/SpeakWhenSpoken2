import asyncio
import json
import base64
import sys

import numpy as np
import sounddevice as sd
import websockets

from config import SAMPLE_RATE, CHUNK_SAMPLES, CHANNELS


async def run(host, port):
    uri = f"ws://{host}:{port}"
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as ws:
        print("Connected. Streaming mic audio. Ctrl+C to stop.")
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()

        def audio_cb(indata, frames, time_info, status):
            loop.call_soon_threadsafe(queue.put_nowait, indata[:, 0].astype(np.float32))

        stream = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SAMPLES,
                                channels=CHANNELS, callback=audio_cb)
        stream.start()

        send = asyncio.create_task(_sender(ws, queue))
        recv = asyncio.create_task(_receiver(ws))
        try:
            await asyncio.gather(send, recv)
        except KeyboardInterrupt:
            await ws.send(json.dumps({"type": "stop"}))
        finally:
            stream.stop()
            stream.close()


async def _sender(ws, queue):
    while True:
        chunk = await queue.get()
        encoded = base64.b64encode(chunk.tobytes()).decode()
        await ws.send(json.dumps({"type": "audio", "data": encoded}))


async def _receiver(ws):
    async for message in ws:
        msg = json.loads(message)

        if "wake_response" in msg:
            wr = msg["wake_response"]
            print(f"\n>>> FAUNA [{wr['source']}]: {wr['text']}")

        elif "gate_open" in msg:
            g = msg["gate_open"]
            decision = g.get("decision")
            source = g.get("source", "?")
            if decision:
                print(f"\n>>> GATE [{source}] (opening={g['ai_opening']:.2f}): {decision}")
            else:
                print(f"\n--- GATE fired, draft stale (opening={g['ai_opening']:.2f})")

        elif "vap" in msg:
            v = msg["vap"]
            s = msg.get("state", {})
            opening = v["ai_opening"]
            silence = s.get("silence_gap_sec", 0)
            if opening > 0.6 or silence > 2.0:
                print(f"  vap: opening={opening:.2f} mode={s.get('mode','?')} silence={silence:.1f}s")

        if "entities" in msg:
            ents = msg["entities"]
            if ents:
                labels = [f"{e['label']}:{e['text']}" for e in ents[:5]]
                print(f"  entities: {', '.join(labels)}")

        if "graph_text" in msg:
            n_lines = msg["graph_text"].count("\n") + 1
            print(f"  graph: {n_lines} facts tracked")


host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
port = int(sys.argv[2]) if len(sys.argv) > 2 else "8765"
asyncio.run(run(host, port))
