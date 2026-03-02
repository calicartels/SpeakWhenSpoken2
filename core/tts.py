import io
import os
import logging

log = logging.getLogger("tts")

client = None
ok = None


def init():
    global client, ok
    if ok is not None:
        return client
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        ok = False
        return None
    from elevenlabs import ElevenLabs
    client = ElevenLabs(api_key=key)
    ok = True
    return client


def is_available():
    if ok is None:
        init()
    return bool(ok)


def synthesize(text, voice=None, model=None):
    import config
    c = init()
    if c is None:
        return None
    voice = voice or config.TTS_VOICE
    model = model or config.TTS_MODEL
    audio_iter = c.text_to_speech.convert(
        text=text, voice_id=voice, model_id=model, output_format="mp3_44100_128",
    )
    buf = io.BytesIO()
    for chunk in audio_iter:
        buf.write(chunk)
    return buf.getvalue()
