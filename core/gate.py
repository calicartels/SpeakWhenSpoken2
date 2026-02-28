import config


def should_open(vap_opening, mode, dominant_prob,
                silence_gap_sec=0.0, turn_hold=0.0):
    if vap_opening is None:
        return False

    signals = {
        "silence_gap": min(silence_gap_sec / 2.0, 1.0),
        "speaker_fading": (
            max(0.0, 1.0 - dominant_prob / config.GATE_FADE_PROB)
            if dominant_prob is not None else 0.0
        ),
        "prosodic_boundary": 1.0 - turn_hold,
        "vap_score": vap_opening,
    }

    composite = sum(config.GATE_WEIGHTS[k] * signals[k] for k in config.GATE_WEIGHTS)
    return composite >= config.GATE_THRESHOLD


def deduplicate(openings):
    if not openings:
        return []
    result = [openings[0]]
    for ho in openings[1:]:
        gap = ho["timestamp"] - result[-1]["timestamp"]
        if gap >= config.GATE_SUPPRESS_SEC:
            result.append(ho)
        elif ho["ai_opening"] > result[-1]["ai_opening"]:
            result[-1] = ho
    return result


def filter_openings(openings):
    deduped = deduplicate(openings)
    for ho in deduped:
        if ho["mode"] == "silence":
            ho["reason"] = "silence_gap"
        elif ho.get("dominant_prob") is not None and ho["dominant_prob"] < config.GATE_FADE_PROB:
            ho["reason"] = "speaker_fading"
        else:
            ho["reason"] = "prosodic_boundary"
    return deduped
