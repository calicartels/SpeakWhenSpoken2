import numpy as np

try:
    from maai import Maai, MaaiInput
    HAS_MAAI = True
except ImportError:
    HAS_MAAI = False

MAAI_FRAME_RATE = 10
SAMPLE_RATE = 16000


def load_vap(device="cpu", lang="en"):
    if not HAS_MAAI:
        print("MaAI not installed, using energy-based fallback")
        return None
    return {"device": device, "lang": lang, "loaded": True}


def predict_from_audio(model, ch1, ch2, source_label):
    if model is not None and HAS_MAAI:
        return _predict_maai(model, ch1, ch2, source_label)
    return _predict_energy(ch1, ch2, source_label)


def _predict_maai(model, ch1, ch2, source_label):
    import os
    import tempfile
    import soundfile as sf

    stereo = np.stack([ch1, ch2], axis=0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, stereo.T, SAMPLE_RATE)
        tmp_path = f.name
    try:
        maai = Maai(
            mode="vap",
            lang=model["lang"],
            frame_rate=MAAI_FRAME_RATE,
            audio_ch1=MaaiInput.File(tmp_path, channel=0),
            audio_ch2=MaaiInput.File(tmp_path, channel=1),
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
        return _format_maai_result(last_result, source_label)
    finally:
        os.unlink(tmp_path)


def _format_maai_result(result, source_label):
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


def _predict_energy(ch1, ch2, source_label):
    n = SAMPLE_RATE * 2
    c1 = ch1[-n:] if len(ch1) >= n else ch1
    c2 = ch2[-n:] if len(ch2) >= n else ch2
    e1 = np.sqrt(np.mean(c1 ** 2) + 1e-10)
    e2 = np.sqrt(np.mean(c2 ** 2) + 1e-10)
    floor = 0.02

    if e1 < floor and e2 < floor:
        is_ai = "ai" in source_label.lower()
        return {
            "source": source_label,
            "p_now_ch1": float(e1 / floor),
            "p_now_ch2": float(e2 / floor),
            "p_future_ch1": 0.2,
            "p_future_ch2": 0.8,
            "ai_opening": 0.8 if is_ai else None,
            "turn_hold": 0.2 if is_ai else None,
            "confidence": 0.6,
        }

    is_ai = "ai" in source_label.lower()
    if is_ai:
        level = float(np.clip(e1 / 0.15, 0, 1))
        p_now_ch1, p_now_ch2 = level, 0.0
        if len(ch1) >= SAMPLE_RATE * 4:
            old = ch1[-(n + SAMPLE_RATE * 2):-n]
            e1_old = np.sqrt(np.mean(old ** 2) + 1e-10)
            trend = e1 / (e1_old + 1e-8)
            p_fut_ch1 = float(np.clip(level * min(trend, 1.5), 0, 1))
        else:
            p_fut_ch1 = level
        p_fut_ch2 = 1.0 - p_fut_ch1
        return {
            "source": source_label,
            "p_now_ch1": p_now_ch1,
            "p_now_ch2": p_now_ch2,
            "p_future_ch1": p_fut_ch1,
            "p_future_ch2": p_fut_ch2,
            "ai_opening": p_fut_ch2,
            "turn_hold": p_fut_ch1,
            "confidence": abs(p_fut_ch1 - p_fut_ch2),
        }

    total = e1 + e2 + 1e-10
    p_now_ch1 = float(np.clip(e1 / total, 0, 1))
    p_now_ch2 = float(np.clip(e2 / total, 0, 1))
    if len(ch1) >= SAMPLE_RATE * 4:
        old = ch1[-(n + SAMPLE_RATE * 2):-n]
        e1_old = np.sqrt(np.mean(old ** 2) + 1e-10)
        trend = e1 / (e1_old + 1e-8) if e1_old > 1e-8 else 1.0
        p_fut_ch1 = float(np.clip(p_now_ch1 * trend, 0, 1))
    else:
        p_fut_ch1 = p_now_ch1
    p_fut_ch2 = 1.0 - p_fut_ch1
    return {
        "source": source_label,
        "p_now_ch1": p_now_ch1,
        "p_now_ch2": p_now_ch2,
        "p_future_ch1": p_fut_ch1,
        "p_future_ch2": p_fut_ch2,
        "ai_opening": p_fut_ch2 if is_ai else None,
        "turn_hold": p_fut_ch1 if is_ai else None,
        "confidence": abs(p_fut_ch1 - p_fut_ch2),
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
