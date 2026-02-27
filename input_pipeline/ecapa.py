import torch
import torchaudio
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier

import config


def load_model():
    model = EncoderClassifier.from_hparams(
        source=config.ECAPA_MODEL,
        savedir="pretrained_models/ecapa",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    return model


def load_audio(path):
    wav, sr = torchaudio.load(path)
    if sr != config.SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, config.SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0)


def segment_audio(wav, segments):
    chunks = []
    for start_sec, end_sec, label in segments:
        dur = end_sec - start_sec
        if dur < config.MIN_SEGMENT_SEC:
            continue
        if dur > config.MAX_SEGMENT_SEC:
            end_sec = start_sec + config.MAX_SEGMENT_SEC
        s = int(start_sec * config.SAMPLE_RATE)
        e = int(end_sec * config.SAMPLE_RATE)
        chunk = wav[s:e]
        if len(chunk) > 0:
            chunks.append({"audio": chunk, "start": start_sec, "end": end_sec, "label": label})
    return chunks


def extract_embeddings(model, chunks):
    embs = []
    for c in chunks:
        wav = c["audio"].unsqueeze(0)
        emb = model.encode_batch(wav).squeeze()
        embs.append(emb.cpu())
    return torch.stack(embs) if embs else torch.empty(0)


def cosine_sim(embs):
    normed = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)
    return (normed @ normed.T).numpy()


def format_similarity(chunks, sim):
    n = len(chunks)
    labels = [f"{c['label']}({c['start']:.1f}s)" for c in chunks]
    header = f"{'':>16}" + "".join(f"{l:>16}" for l in labels)
    lines = [f"Cosine similarity matrix ({config.ECAPA_EMB_DIM}-dim embeddings):", header]
    for i in range(n):
        row = f"{labels[i]:>16}"
        for j in range(n):
            row += f"{sim[i, j]:>16.3f}"
        lines.append(row)
    lines.extend([
        "",
        "> 0.85: same speaker | 0.60-0.85: likely same | < 0.60: different",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    model = load_model()
    wav = load_audio(config.TEST_AUDIO)
    duration = len(wav) / config.SAMPLE_RATE
    segments = [
        (0.0, 4.0, "spk0"),
        (4.5, 9.0, "spk1"),
        (10.0, 14.0, "spk2"),
    ]
    segments = [(s, min(e, duration), l) for s, e, l in segments if s < duration]
    chunks = segment_audio(wav, segments)
    embs = extract_embeddings(model, chunks)
    if len(embs) == 0:
        print("No valid segments. Check segment times vs audio duration.")
    else:
        sim = cosine_sim(embs)
        print(format_similarity(chunks, sim))
        torch.save({"embeddings": embs, "chunks": chunks}, "test_audio/ecapa_embeddings.pt")
