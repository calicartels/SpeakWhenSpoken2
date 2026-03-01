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
    """No local model -- vLLM serves Voxtral externally."""
    return None, None


async def new_stream(_model=None, _proc=None):
    """Open a realtime session with the vLLM WebSocket endpoint."""
    ws = None
    for attempt in range(MAX_CONNECT_ATTEMPTS):
        try:
            ws = await asyncio.wait_for(
                websockets.connect(VLLM_URL), timeout=15
            )
            break
        except Exception as e:
            if attempt < MAX_CONNECT_ATTEMPTS - 1:
                log.info("vLLM not ready (attempt %d/%d): %s", attempt + 1, MAX_CONNECT_ATTEMPTS, e)
                await asyncio.sleep(CONNECT_RETRY_SEC)

    if ws is None:
        log.error(f"Cannot connect to vLLM at {VLLM_URL} after {MAX_CONNECT_ATTEMPTS} attempts")
        return None

    response = json.loads(await ws.recv())
    if response.get("type") != "session.created":
        raise RuntimeError(f"Expected session.created, got {response}")

    await ws.send(json.dumps({"type": "session.update", "model": config.VOXTRAL_MODEL}))
    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    stream = {"ws": ws, "text_buf": [], "done": False}
    stream["recv_task"] = asyncio.create_task(_recv_loop(stream))
    return stream


async def _recv_loop(stream):
    """Background coroutine: drain transcription deltas from vLLM."""
    try:
        async for message in stream["ws"]:
            data = json.loads(message)
            if data["type"] in ("transcription.delta", "response.text.delta"):
                stream["text_buf"].append(data.get("delta", ""))
            elif data["type"] in ("transcription.done", "response.done"):
                stream["done"] = True
            elif data["type"] == "error":
                log.warning("vLLM error: %s", data.get("error", data))
    except websockets.exceptions.ConnectionClosed:
        pass
    except asyncio.CancelledError:
        pass


async def feed_audio(stream, samples_f32):
    """Convert float32 samples to PCM16 and send to vLLM."""
    pcm16 = (samples_f32 * 32767).astype(np.int16)
    audio_b64 = base64.b64encode(pcm16.tobytes()).decode()
    try:
        await stream["ws"].send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        }))
    except websockets.exceptions.ConnectionClosed:
        pass


def get_text(stream):
    """Non-blocking: drain all buffered transcription text."""
    if not stream["text_buf"]:
        return ""
    text = "".join(stream["text_buf"])
    stream["text_buf"].clear()
    return text


async def stop_stream(stream):
    """Signal end-of-audio and close the vLLM session."""
    try:
        await stream["ws"].send(json.dumps({
            "type": "input_audio_buffer.commit", "final": True,
        }))
    except Exception:
        pass
    stream["recv_task"].cancel()
    try:
        await stream["ws"].close()
    except Exception:
        pass
