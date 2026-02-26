def new_state(session_id=0):
    return {
        "timestamp": 0.0,
        "session_id": session_id,
        "speakers": {},
        "dyad": {
            "mode": "silence",
            "active_pair": None,
            "dominant_speaker": None,
            "dominant_identity": None,
        },
        "vap": {
            "ai_opening": 0.0,
            "turn_hold": 0.0,
            "confidence": 0.0,
            "source": None,
        },
        "pair_history": {},
        "transcript": [],
        "silence": {
            "current_gap_sec": 0.0,
            "last_gap_start": None,
            "last_gap_duration": 0.0,
            "longest_gap_sec": 0.0,
        },
        "session": {
            "session_id": session_id,
            "started_at": 0.0,
            "duration_sec": 0.0,
            "total_speakers_seen": 0,
            "fifth_speaker_detected": False,
        },
    }


def update_speakers(state, probs, timestamp):
    state["timestamp"] = timestamp
    state["session"]["duration_sec"] = timestamp - state["session"]["started_at"]
    any_active = False

    for slot_id, prob in enumerate(probs):
        if slot_id not in state["speakers"] and prob > 0.05:
            state["speakers"][slot_id] = {
                "identity": None,
                "total_speaking_sec": 0.0,
                "last_spoke_at": None,
                "is_active": False,
                "current_prob": 0.0,
            }
            state["session"]["total_speakers_seen"] = len(state["speakers"])
        if slot_id not in state["speakers"]:
            continue
        spk = state["speakers"][slot_id]
        was_active = spk["is_active"]
        spk["is_active"] = prob > 0.3
        spk["current_prob"] = prob
        if spk["is_active"]:
            spk["total_speaking_sec"] += 0.08
            spk["last_spoke_at"] = None
            any_active = True
        elif was_active and not spk["is_active"]:
            spk["last_spoke_at"] = timestamp

    if any_active:
        if state["silence"]["current_gap_sec"] > 0:
            gap = state["silence"]["current_gap_sec"]
            state["silence"]["last_gap_duration"] = gap
            if gap > state["silence"]["longest_gap_sec"]:
                state["silence"]["longest_gap_sec"] = gap
            state["silence"]["current_gap_sec"] = 0.0
    else:
        if state["silence"]["current_gap_sec"] == 0.0:
            state["silence"]["last_gap_start"] = timestamp
        state["silence"]["current_gap_sec"] += 0.08


def update_dyad(state, dyad_out):
    state["dyad"]["mode"] = dyad_out["mode"]
    state["dyad"]["active_pair"] = dyad_out["active_pair"]
    state["dyad"]["dominant_speaker"] = dyad_out["dominant"]
    if dyad_out["dominant"] is not None and dyad_out["dominant"] in state["speakers"]:
        state["dyad"]["dominant_identity"] = state["speakers"][dyad_out["dominant"]]["identity"]
    else:
        state["dyad"]["dominant_identity"] = None
    pair = dyad_out["active_pair"]
    if pair is not None:
        if pair not in state["pair_history"]:
            state["pair_history"][pair] = {"total_sec": 0.0, "turn_count": 0, "last_active": None}
        ph = state["pair_history"][pair]
        ph["total_sec"] += 0.08
        ph["last_active"] = state["timestamp"]


def update_vap(state, vap_out):
    state["vap"]["ai_opening"] = vap_out["ai_opening"] if vap_out["ai_opening"] is not None else 0.5
    state["vap"]["turn_hold"] = vap_out["turn_hold"] if vap_out["turn_hold"] is not None else 0.5
    state["vap"]["confidence"] = vap_out["confidence"]
    state["vap"]["source"] = vap_out["source"]


def add_transcript(state, speaker, slot_id, start, end, text):
    state["transcript"].append({"speaker": speaker, "slot_id": slot_id, "start": start, "end": end, "text": text})
    cutoff = state["timestamp"] - 60.0
    state["transcript"] = [t for t in state["transcript"] if t["end"] > cutoff]


def set_identity(state, slot_id, identity):
    if slot_id in state["speakers"]:
        state["speakers"][slot_id]["identity"] = identity


def increment_pair_turns(state, pair):
    if pair in state["pair_history"]:
        state["pair_history"][pair]["turn_count"] += 1


def render_for_llm(state):
    lines = [f"MEETING STATE at {state['timestamp']:.2f}s:"]
    spk_parts = []
    for slot_id, spk in sorted(state["speakers"].items()):
        name = spk["identity"] or f"slot_{slot_id}"
        if spk["is_active"]:
            spk_parts.append(f"{name} (speaking)")
        elif spk["last_spoke_at"] is not None:
            gap = state["timestamp"] - spk["last_spoke_at"]
            spk_parts.append(f"{name} (silent {gap:.1f}s)")
        else:
            spk_parts.append(f"{name} (never spoke)")
    lines.append("Speakers: " + ", ".join(spk_parts))
    d = state["dyad"]
    dom_name = d["dominant_identity"] or (f"slot_{d['dominant_speaker']}" if d["dominant_speaker"] is not None else "nobody")
    lines.append(f"Current: {d['mode']}, dominant={dom_name}")
    v = state["vap"]
    ao = v["ai_opening"] if v["ai_opening"] is not None else 0.5
    th = v["turn_hold"] if v["turn_hold"] is not None else 0.5
    lines.append(f"VAP: ai_opening={ao:.2f}, turn_hold={th:.2f}, conf={v['confidence']:.2f}")
    if state["silence"]["current_gap_sec"] > 0:
        lines.append(f"Silence: {state['silence']['current_gap_sec']:.1f}s ongoing")
    lines.append("\nRecent transcript:")
    for t in state["transcript"][-5:]:
        name = t["speaker"] or f"slot_{t['slot_id']}"
        lines.append(f"  [{t['start']:.1f}-{t['end']:.1f}] {name}: {t['text']}")
    return "\n".join(lines)
