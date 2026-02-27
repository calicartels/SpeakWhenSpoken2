import time

from gliner import GLiNER

import config

PERSON_STOPWORDS = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "it", "its", "itself",
    "who", "whom", "whose",
    "someone", "anyone", "everyone", "nobody",
    "people", "person", "model",
}


def load_model():
    return GLiNER.from_pretrained(config.GLINER_MODEL)


def extract(model, text):
    if not text or len(text.strip()) < 10:
        return []
    entities = model.predict_entities(text, config.ENTITY_LABELS, threshold=config.ENTITY_THRESHOLD)
    filtered = []
    for e in entities:
        if e["label"] == "person" and e["text"].lower().strip() in PERSON_STOPWORDS:
            continue
        filtered.append({"text": e["text"], "label": e["label"], "score": e["score"]})
    return filtered


def new_store():
    return {"entities": {}, "segments": [], "speakers": {}}


def update(store, segment, entities):
    speaker = segment.get("speaker", f"slot_{segment.get('slot_id', '?')}")
    store["segments"].append({
        "speaker": speaker,
        "start": segment.get("start", 0),
        "end": segment.get("end", 0),
        "entity_count": len(entities),
    })

    if speaker not in store["speakers"]:
        store["speakers"][speaker] = {"total_segments": 0, "total_sec": 0.0, "mentioned_entities": []}
    spk = store["speakers"][speaker]
    spk["total_segments"] += 1
    spk["total_sec"] += segment.get("end", 0) - segment.get("start", 0)

    for ent in entities:
        key = (ent["label"], ent["text"].lower())
        if key not in store["entities"]:
            store["entities"][key] = {
                "label": ent["label"],
                "text": ent["text"],
                "count": 0,
                "first_seen": segment.get("start", 0),
                "last_seen": segment.get("start", 0),
                "mentioned_by": [],
                "best_score": 0.0,
            }
        entry = store["entities"][key]
        entry["count"] += 1
        entry["last_seen"] = segment.get("start", 0)
        if speaker not in entry["mentioned_by"]:
            entry["mentioned_by"].append(speaker)
        entry["best_score"] = max(entry["best_score"], ent["score"])
        spk["mentioned_entities"].append(ent["text"])


def extract_all(model, transcript):
    store = new_store()
    t0 = time.time()
    for seg in transcript:
        entities = extract(model, seg.get("text", ""))
        update(store, seg, entities)
    return store, time.time() - t0


def render_for_llm(store, max_entities=30):
    lines = ["MEETING MEMORY:"]
    if store["speakers"]:
        parts = []
        for name, spk in sorted(store["speakers"].items(), key=lambda x: -x[1]["total_sec"]):
            parts.append(f"{name} ({spk['total_sec']:.0f}s, {spk['total_segments']} segments)")
        lines.append("Speakers: " + ", ".join(parts))

    by_label = {}
    for (label, _), entry in store["entities"].items():
        by_label.setdefault(label, []).append(entry)

    for label in config.ENTITY_LABELS:
        if label not in by_label:
            continue
        entries = sorted(by_label[label], key=lambda x: -x["count"])[:max_entities]
        texts = [f"{e['text']} (x{e['count']})" if e["count"] > 1 else e["text"] for e in entries]
        lines.append(f"{label}: {', '.join(texts)}")
    return "\n".join(lines)


def format_stats(store, elapsed):
    n_ents = len(store["entities"])
    n_segs = len(store["segments"])
    by_label = {}
    for (label, _) in store["entities"]:
        by_label[label] = by_label.get(label, 0) + 1

    lines = [f"Extracted {n_ents} entities from {n_segs} segments in {elapsed:.2f}s"]
    for label, count in sorted(by_label.items(), key=lambda x: -x[1]):
        lines.append(f"  {label}: {count}")
    return "\n".join(lines)
