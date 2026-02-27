import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from INPUT_PIPELINE import transcribe
from VAP import orchestrate, vap
from LLM import memory, decide
from tests import baseline


def run():
    probs = np.load(config.PROBS_CACHE_PATH).tolist()
    audio, sr = sf.read(config.TEST_AUDIO_20MIN, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    print(f"Audio: {len(audio)/sr:.1f}s, {len(probs)} frames")

    # Phase 1.1: baseline comparison
    print(f"\n{'=' * 60}\nPHASE 1.1: BASELINE\n{'=' * 60}")
    raw_base, base_openings = baseline.compute_baseline(probs)
    print(f"Baseline: {len(raw_base)} raw -> {len(base_openings)} gated")

    vap_model = vap.load_vap()
    _, _, vap_openings = orchestrate.process_file(audio, sr, probs, vap_model)

    print(baseline.compare_at_timestamps(probs, vap_openings))
    print(baseline.format_summary(vap_openings, base_openings))

    # Phase 1.2: memory extraction
    print(f"\n{'=' * 60}\nPHASE 1.2: MEMORY\n{'=' * 60}")
    segments = transcribe.extract_segments(probs)
    print(f"{len(segments)} segments")

    if os.path.exists(config.TRANSCRIPT_CACHE_PATH):
        with open(config.TRANSCRIPT_CACHE_PATH) as f:
            transcript = json.load(f)
        print(f"Loaded cached transcript: {len(transcript)} segments")
    else:
        from INPUT_PIPELINE.voxtral import load_model as load_voxtral
        model, proc = load_voxtral()
        transcript = transcribe.transcribe_all(audio, sr, segments, model, proc)
        with open(config.TRANSCRIPT_CACHE_PATH, "w") as f:
            json.dump(transcript, f)
        print(f"Transcribed and cached {len(transcript)} segments")

    gliner = memory.load_model()
    store, elapsed = memory.extract_all(gliner, transcript)
    print(memory.format_stats(store, elapsed))
    print(f"\n{memory.render_for_llm(store)}")

    # Phase 1.3: LLM decisions
    if not os.environ.get("OPEN_ROUTER_API_KEY"):
        print(f"\n{'=' * 60}\nPHASE 1.3: SKIPPED (no OPEN_ROUTER_API_KEY)\n{'=' * 60}")
        return

    print(f"\n{'=' * 60}\nPHASE 1.3: LLM DECISIONS\n{'=' * 60}")
    memory_str = memory.render_for_llm(store)
    state_str = "MEETING STATE: offline validation (see transcript context below)"

    results = decide.decide_batch(state_str, transcript, memory_str, vap_openings)
    print(f"\n{decide.format_results(results)}")


run()
