import os
import tempfile

import numpy as np
import soundfile as sf
from maai import Maai, MaaiInput

FRAME_RATE = 10
SAMPLE_RATE = 16000


def load_vap(device="cpu", lang="en"):
    return {"device": device, "lang": lang}


def predict_from_audio(model, ch1, ch2, source_label):
    ch1_path = _write_temp_wav(ch1)
    ch2_path = _write_temp_wav(ch2)
    try:
        wav1 = MaaiInput.Wav(wav_file_path=ch1_path)
        wav2 = MaaiInput.Wav(wav_file_path=ch2_path)
        maai = Maai(
            mode="vap",
            lang=model["lang"],
            frame_rate=FRAME_RATE,
            audio_ch1=wav1,
            audio_ch2=wav2,
            device=model["device"],
        )
        maai.start()
        last_result = None
        while True:
            result = maai.get_result()
            if result is None:
                break
            last_result = result
        if last_result is None:
            return _empty_output(source_label)
        return _format_result(last_result, source_label)
    finally:
        os.unlink(ch1_path)
        os.unlink(ch2_path)


def _write_temp_wav(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, SAMPLE_RATE)
        return f.name


def _format_result(result, source_label):
    p_now_0 = float(result.p_now[0])
    p_now_1 = float(result.p_now[1])
    p_fut_0 = float(result.p_future[0])
    p_fut_1 = float(result.p_future[1])
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
