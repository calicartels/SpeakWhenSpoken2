ACTIVE_THRESHOLD = 0.3


def detect(probs, timestamp):
    active = [(i, p) for i, p in enumerate(probs) if p > ACTIVE_THRESHOLD]
    active.sort(key=lambda x: x[1], reverse=True)

    if len(active) == 0:
        return {"timestamp": timestamp, "mode": "silence", "active_pair": None, "dominant": None, "probs": {}}
    if len(active) == 1:
        return {
            "timestamp": timestamp,
            "mode": "solo",
            "active_pair": None,
            "dominant": active[0][0],
            "probs": {active[0][0]: active[0][1]},
        }
    if len(active) == 2:
        pair = tuple(sorted([active[0][0], active[1][0]]))
        return {
            "timestamp": timestamp,
            "mode": "dyad",
            "active_pair": pair,
            "dominant": active[0][0],
            "probs": {s: p for s, p in active},
        }
    return {
        "timestamp": timestamp,
        "mode": "multi",
        "active_pair": None,
        "dominant": active[0][0],
        "probs": {s: p for s, p in active},
    }


def classify_transition(prev, curr):
    if prev is None:
        return None
    if prev["active_pair"] != curr["active_pair"]:
        return {"type": "dyad_shift", "from_pair": prev["active_pair"], "to_pair": curr["active_pair"], "timestamp": curr["timestamp"]}
    if prev["dominant"] != curr["dominant"]:
        return {"type": "dominant_shift", "pair": curr["active_pair"], "from_speaker": prev["dominant"], "to_speaker": curr["dominant"], "timestamp": curr["timestamp"]}
    return None
