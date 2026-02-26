import torch
import torchaudio

import config
from ecapa import load_model as load_ecapa
from sortformer import diarize, load_model as load_sortformer
from voxtral import load_model as load_voxtral, transcribe_chunk
from identity import (
    load_address_book, new_session, update_slot,
    resolve_identities, get_identity, commit_session,
)


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


def build_identities(segments, embeddings, book):
    session = new_session()

    for i, (seg, emb) in enumerate(zip(segments, embeddings)):
        if emb is None:
            continue
        duration = seg["end"] - seg["start"]
        update_slot(session, seg["speaker"], emb, duration)

    resolve_identities(session, book)

    segment_ids = {}
    for i, seg in enumerate(segments):
        segment_ids[i] = get_identity(session, seg["speaker"])

    return segment_ids, session


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


def format_session_summary(session):
    lines = ["\nSESSION IDENTITY SUMMARY", "-" * 40]
    for slot_id, slot in session["slots"].items():
        name = slot["identity"] or "(unresolved)"
        lines.append(f"  {slot_id} -> {name} ({slot['total_sec']:.1f}s accumulated)")
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

book = load_address_book()
identities, session = build_identities(segments, embeddings, book)
commit_session(session, book)

transcripts = []
for seg in segments:
    s = int(seg["start"] * sr)
    e = int(seg["end"] * sr)
    chunk = wav[s:e].numpy()
    text = transcribe_chunk(voxtral_model, voxtral_proc, chunk, sr)
    transcripts.append(text)

print(format_output(segments, identities, transcripts))
print(format_session_summary(session))
print(f"\n{len(segments)} segments, {len(set(identities.values()))} speakers, {duration:.1f}s")