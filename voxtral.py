import os

import torch
from mistral_common.tokens.tokenizers.audio import Audio
from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

import config


def load_model():
    processor = AutoProcessor.from_pretrained(config.VOXTRAL_MODEL)
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        config.VOXTRAL_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if torch.cuda.is_available():
        gb = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM: {gb:.1f}GB")
    return model, processor


def transcribe(model, processor, audio_path):
    audio = Audio.from_file(audio_path, strict=False)
    audio.resample(processor.feature_extractor.sampling_rate)
    inputs = processor(audio.audio_array, return_tensors="pt")
    inputs = inputs.to(model.device, dtype=model.dtype)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return text


def transcribe_chunk(model, processor, chunk, sr):
    """chunk: numpy array, sr: sample rate. Skips segments < 0.3s."""
    if len(chunk) < sr * 0.3:
        return ""
    inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
    inputs = inputs.to(model.device, dtype=model.dtype)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return text.strip()


if __name__ == "__main__":
    model, processor = load_model()
    print(transcribe(model, processor, config.TEST_AUDIO))
    if os.path.exists(config.HARD_AUDIO):
        print(transcribe(model, processor, config.HARD_AUDIO))
