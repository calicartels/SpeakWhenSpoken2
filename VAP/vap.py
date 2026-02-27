import queue

import numpy as np
from maai import Maai, MaaiInput

FRAME_RATE = 10
SAMPLE_RATE = 16000


def load_vap(device="cpu", lang="en"):
    ch1 = MaaiInput.Chunk()
    ch2 = MaaiInput.Chunk()
    maai = Maai(
        mode="vap", lang=lang, frame_rate=FRAME_RATE,
        audio_ch1=ch1, audio_ch2=ch2, device=device,
    )
    return {"maai": maai}


def push_frame(model, frame_audio):
    maai = model["maai"]
    audio = np.asarray(frame_audio, dtype=np.float32)
    zeros = np.zeros_like(audio)
    maai.process(audio, zeros)


def get_latest(model, source_label):
    maai = model["maai"]
    last = None
    while True:
        try:
            r = maai.result_dict_queue.get_nowait()
            if r is not None:
                last = r
        except queue.Empty:
            break
    if last is None:
        return _empty_output(source_label)
    return _format_result(last, source_label)


def _format_result(result, source_label):
    p_now_0 = float(result["p_now"][0])
    p_now_1 = float(result["p_now"][1])
    p_fut_0 = float(result["p_future"][0])
    p_fut_1 = float(result["p_future"][1])
    is_ai = "ai" in source_label.lower()
    return {
        "source": source_label,
        "p_now_ch1": p_now_0,
        "p_now_ch2": p_now_1,
        "p_future_ch1": p_fut_0,
        "p_future_ch2": p_fut_1,
        "ai_opening": p_fut_1 if is_ai else None,
        "turn_hold": p_fut_0 if is_ai else None,
        "confidence": abs(p_fut_0 - p_fut_1),
    }


def _empty_output(source_label):
    is_ai = "ai" in source_label.lower()
    return {
        "source": source_label,
        "p_now_ch1": 0.0,
        "p_now_ch2": 0.0,
        "p_future_ch1": 0.5,
        "p_future_ch2": 0.5,
        "ai_opening": 0.5 if is_ai else None,
        "turn_hold": 0.5 if is_ai else None,
        "confidence": 0.0,
    }
