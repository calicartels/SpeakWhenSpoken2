import asyncio
import base64
import json
import logging

import numpy as np
import websockets

import config

log = logging.getLogger("voxtral")

VLLM_URL = f"ws://localhost:{config.VLLM_PORT}/v1/realtime"
MAX_CONNECT_ATTEMPTS = 60
CONNECT_RETRY_SEC = 2


def load_model():
    return None, None


async def new_stream(model=None, proc=None):
    ws = None
    for attempt in range(MAX_CONNECT_ATTEMPTS):
        ws = await asyncio.wait_for(
            websockets.connect(VLLM_URL), timeout=15
        )
        break

    if ws is None:
        return None

    response = json.loads(await ws.recv())
    if response.get("type") != "session.created":
        raise RuntimeError(f"Expected session.created, got {response}")

    await ws.send(json.dumps({"type": "session.update", "model": config.VOXTRAL_MODEL}))
    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    stream = {"ws": ws, "text_buf": [], "done": False}
    stream["recv_task"] = asyncio.create_task(recv_loop(stream))
    return stream


async def recv_loop(stream):
    async for message in stream["ws"]:
        data = json.loads(message)
        if data["type"] in ("transcription.delta", "response.text.delta"):
            stream["text_buf"].append(data.get("delta", ""))
        elif data["type"] in ("transcription.done", "response.done"):
            stream["done"] = True


async def feed_audio(stream, samples_f32):
    pcm16 = (samples_f32 * 32767).astype(np.int16)
    audio_b64 = base64.b64encode(pcm16.tobytes()).decode()
    await stream["ws"].send(json.dumps({
        "type": "input_audio_buffer.append",
        "audio": audio_b64,
    }))


def get_text(stream):
    if not stream["text_buf"]:
        return ""
    text = "".join(stream["text_buf"])
    stream["text_buf"].clear()
    return text


async def stop_stream(stream):
    await stream["ws"].send(json.dumps({
        "type": "input_audio_buffer.commit", "final": True,
    }))
    stream["recv_task"].cancel()
    await stream["ws"].close()
