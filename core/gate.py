import config


def should_open(vap_opening, mode, dominant_prob,
                silence_gap_sec=0.0, turn_hold=0.0):
    """Simplified gate: VAP + silence threshold.

    Opens if:
      - VAP opening >= threshold AND silence >= min silence
      - OR silence alone >= force-open threshold (long pause)
    """
    if vap_opening is None:
        return False

    # Long silence always opens (someone should speak)
    if silence_gap_sec >= config.GATE_SILENCE_FORCE:
        return True

    # VAP says AI should speak AND there's a natural gap
    if vap_opening >= config.GATE_THRESHOLD and silence_gap_sec >= config.GATE_SILENCE_MIN:
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
        if ho.get("silence_gap_sec", 0) >= config.GATE_SILENCE_FORCE:
            ho["reason"] = "long_silence"
        elif ho.get("mode") == "silence":
            ho["reason"] = "silence_gap"
        else:
            ho["reason"] = "vap_opening"
    return deduped
