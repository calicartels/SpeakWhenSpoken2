# integrate.py — Sortformer + ECAPA + Voxtral → speaker-attributed transcript

import torch
import torchaudio

import config
from ecapa import load_model as load_ecapa
from sortformer import diarize, load_model as load_sortformer
from voxtral import load_model as load_voxtral, transcribe_chunk


def parse_segments(raw_segments):
    parsed = []
    for seg in raw_segments:
        parts = str(seg).split()
        if len(parts) >= 3:
            parsed.append({
                "start": float(parts[0]),
                "end": float(parts[1]),
                "speaker": parts[2],
            })
    return parsed


def extract_embedding(ecapa, wav, start_sec, end_sec, sr):
    s = int(start_sec * sr)
    e = int(end_sec * sr)
    chunk = wav[s:e]
    if len(chunk) < sr * 0.5:
        return None
    emb = ecapa.encode_batch(chunk.unsqueeze(0)).squeeze().cpu()
    return emb


def cosine(a, b):
    return torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)


def assign_identities(segments, embeddings, threshold=0.45):
    identities = {}
    segment_ids = {}
    next_id = 0
    for i, emb in enumerate(embeddings):
        if emb is None:
            segment_ids[i] = "unknown"
            continue
        matched = False
        for identity_id, stored_embs in identities.items():
            avg_emb = torch.stack(stored_embs).mean(dim=0)
            sim = cosine(emb, avg_emb).item()
            if sim > threshold:
                identities[identity_id].append(emb)
                segment_ids[i] = identity_id
                matched = True
                break
        if not matched:
            identity_id = f"person_{next_id}"
            identities[identity_id] = [emb]
            segment_ids[i] = identity_id
            next_id += 1
    return segment_ids


def format_output(segments, identities, transcripts):
    lines = ["SPEAKER-ATTRIBUTED TRANSCRIPT", "=" * 70]
    for i, seg in enumerate(segments):
        identity = identities.get(i, "unknown")
        text = transcripts[i]
        if not text:
            continue
        lines.append(f"\n[{seg['start']:6.2f}s - {seg['end']:6.2f}s] {identity} ({seg['speaker']})")
        lines.append(f"  \"{text}\"")
    return "\n".join(lines)


sortformer = load_sortformer()
ecapa = load_ecapa()
voxtral_model, voxtral_proc = load_voxtral()

if torch.cuda.is_available():
    print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

wav, sr = torchaudio.load(config.HARD_AUDIO)
wav = wav.squeeze(0)
if sr != config.SAMPLE_RATE:
    wav = torchaudio.functional.resample(wav, sr, config.SAMPLE_RATE)
    sr = config.SAMPLE_RATE
duration = len(wav) / sr

segments_raw, _ = diarize(sortformer, config.HARD_AUDIO)
segments = parse_segments(segments_raw)

embeddings = []
for seg in segments:
    emb = extract_embedding(ecapa, wav, seg["start"], seg["end"], sr)
    embeddings.append(emb)
identities = assign_identities(segments, embeddings)

transcripts = []
for seg in segments:
    s = int(seg["start"] * sr)
    e = int(seg["end"] * sr)
    chunk = wav[s:e].numpy()
    text = transcribe_chunk(voxtral_model, voxtral_proc, chunk, sr)
    transcripts.append(text)

print(format_output(segments, identities, transcripts))
print(f"\n{len(segments)} segments, {len(set(identities.values()))} speakers, {duration:.1f}s")
