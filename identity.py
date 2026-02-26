import os
import torch
import config


def load_address_book():
    if os.path.exists(config.ADDRESS_BOOK_PATH):
        return torch.load(config.ADDRESS_BOOK_PATH, weights_only=True)
    return {}


def save_address_book(book):
    os.makedirs(os.path.dirname(config.ADDRESS_BOOK_PATH), exist_ok=True)
    torch.save(book, config.ADDRESS_BOOK_PATH)


def cosine(a, b):
    return torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)


def new_session():
    return {
        "slots": {},
        "next_anon": 0,
    }


def update_slot(session, slot_id, emb, duration_sec):
    if slot_id not in session["slots"]:
        session["slots"][slot_id] = {
            "emb_sum": torch.zeros(config.ECAPA_EMB_DIM),
            "total_sec": 0.0,
            "identity": None,
        }
    slot = session["slots"][slot_id]
    slot["emb_sum"] += emb * duration_sec
    slot["total_sec"] += duration_sec


def get_slot_average(session, slot_id):
    if slot_id not in session["slots"]:
        return None
    slot = session["slots"][slot_id]
    if slot["total_sec"] < 1e-6:
        return None
    return slot["emb_sum"] / slot["total_sec"]


def lookup_identity(session, slot_id, book):
    avg = get_slot_average(session, slot_id)
    if avg is None:
        return None
    slot = session["slots"][slot_id]
    if slot["total_sec"] < config.IDENTITY_MIN_SEC:
        return None

    best_name, best_sim = None, -1.0
    for name, stored_emb in book.items():
        sim = cosine(avg, stored_emb).item()
        if sim > best_sim:
            best_name, best_sim = name, sim

    if best_sim > config.IDENTITY_THRESHOLD:
        return best_name
    return None


def resolve_identities(session, book):
    for slot_id, slot in session["slots"].items():
        if slot["identity"] is not None:
            continue
        name = lookup_identity(session, slot_id, book)
        if name:
            slot["identity"] = name
        elif slot["total_sec"] >= config.IDENTITY_MIN_SEC:
            slot["identity"] = f"person_{session['next_anon']}"
            session["next_anon"] += 1


def get_identity(session, slot_id):
    if slot_id in session["slots"] and session["slots"][slot_id]["identity"]:
        return session["slots"][slot_id]["identity"]
    return slot_id


def commit_session(session, book):
    for slot_id, slot in session["slots"].items():
        avg = get_slot_average(session, slot_id)
        if avg is None or slot["total_sec"] < config.IDENTITY_MIN_SEC:
            continue
        name = slot["identity"] or slot_id
        if name in book:
            old = book[name]
            w = min(slot["total_sec"] / 30.0, 0.5)
            book[name] = old * (1.0 - w) + avg * w
        else:
            book[name] = avg
    save_address_book(book)


def detect_fifth_speaker(session, emb, book):
    for slot_id in session["slots"]:
        avg = get_slot_average(session, slot_id)
        if avg is not None:
            sim = cosine(emb, avg).item()
            if sim > config.FIFTH_SPEAKER_THRESHOLD:
                return False
    for name, stored in book.items():
        sim = cosine(emb, stored).item()
        if sim > config.FIFTH_SPEAKER_THRESHOLD:
            return False
    return True