import config


def should_open(vap_opening, mode, dominant_prob):
    if vap_opening is None:
        return False
    if mode == "silence" and vap_opening >= config.GATE_THRESHOLD_SILENCE:
        return True
    if mode in ("solo", "dyad") and vap_opening >= config.GATE_THRESHOLD_SPEECH:
        return True
    if dominant_prob is not None and dominant_prob < config.GATE_FADE_PROB:
        if vap_opening >= config.GATE_THRESHOLD_SILENCE:
            return True
    return False


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
