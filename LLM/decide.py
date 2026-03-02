import json
import os

from dotenv import load_dotenv
from openai import OpenAI

import config

load_dotenv()

SYSTEM_PROMPT = """You are an AI participant in a multi-person meeting. You receive:
1. Meeting state (who's speaking, conversation mode, VAP signal)
2. Recent transcript (last few segments around the current moment)
3. Memory store (structured facts extracted from the meeting so far)

You must decide: should you speak right now?

RULES:
- Default is SILENT. You should speak in at most 1-2 out of every 10 openings.
- Stay silent if someone was just asked a direct question (let THEM answer).
- Stay silent if the opening is a rhetorical pause ("right?", "you know?").
- Stay silent if you have nothing genuinely useful to add.
- Stay silent if someone is mid-thought or mid-explanation.
- A silence gap does NOT mean it's your turn. It often means someone is thinking.
- Speak only if: you are EXPLICITLY addressed by name, OR there is a long (>5s)
  silence where no one was asked anything and you have a specific contribution.
- If you speak, keep it to 1-2 sentences MAX unless asked for more.
- Never say "that's interesting", "great point", or other filler.
- Never ask someone to elaborate unless you have a specific reason.

EXAMPLES OF CORRECT SILENCE:
- Host says "Evan, what do you think?" then silence -> SILENT (Evan's turn)
- Speaker trails off with "right?" -> SILENT (rhetorical, not asking you)
- Speaker finishes a point, brief pause -> SILENT (someone else will respond)
- Two speakers going back and forth -> SILENT (you'd be interrupting)

EXAMPLES OF CORRECT SPEAKING:
- Host says "Does our AI assistant have any thoughts on this?" -> SPEAK
- 8 seconds of complete silence, no one was asked anything -> maybe SPEAK
- Someone states an incorrect fact you can correct -> SPEAK (briefly)

Respond with JSON only:
{"speak": true/false, "response": "what you'd say (empty if silent)", "reason": "why"}"""


def get_client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY env var (get one at openrouter.ai/keys)")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def build_context(state_str, transcript_context, memory_str):
    parts = [state_str]
    if transcript_context:
        parts.append("\nRECENT TRANSCRIPT:")
        for t in transcript_context:
            name = t.get("speaker", f"slot_{t.get('slot_id', '?')}")
            parts.append(f"  [{t['start']:.1f}-{t['end']:.1f}] {name}: {t['text']}")
    if memory_str:
        parts.append(f"\n{memory_str}")
    return "\n".join(parts)


def decide(state_str, transcript_context, memory_str, opening):
    ts = opening["timestamp"]
    reason = opening.get("reason", "unknown")
    score = opening["ai_opening"]

    context = build_context(state_str, transcript_context, memory_str)
    user_msg = (
        f"Gate fired at {ts:.1f}s (reason={reason}, score={score:.2f}).\n"
        f"Active speakers: {opening.get('active_speakers', [])}\n"
        f"Mode: {opening.get('mode', 'unknown')}\n\n"
        f"{context}\n\n"
        f"Should you speak? Respond with JSON only."
    )

    model = config.DECISION_MODEL
    if config.DECISION_USE_NITRO:
        model = f"{model}:nitro"

    client = get_client()
    response = client.chat.completions.create(
        model=model,
        max_tokens=config.DECISION_MAX_TOKENS,
        temperature=config.DECISION_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


def decide_batch(state_str, transcript, memory_str, openings):
    results = []
    for ho in openings:
        ts = ho["timestamp"]
        nearby = [t for t in transcript
                  if t["end"] > ts - config.TRANSCRIPT_WINDOW_SEC
                  and t["start"] < ts + 1.0]
        state = state_str() if callable(state_str) else state_str
        decision = decide(state, nearby, memory_str, ho)
        results.append((ho, decision))
        m, s = int(ts // 60), ts % 60
        speak = decision.get("speak", False)
        tag = "SPEAK" if speak else "SILENT"
        print(f"  [{m}:{s:04.1f}] {tag} | {decision.get('reason', '')}")
        if speak:
            print(f"         -> {decision.get('response', '')}")
    return results


def format_results(results):
    lines = [f"{'Time':>8}  {'Decision':>8}  Reason"]
    lines.append("-" * 60)
    speak_count = 0
    for ho, dec in results:
        ts = ho["timestamp"]
        m, s = int(ts // 60), ts % 60
        speak = dec.get("speak", False)
        if speak:
            speak_count += 1
        tag = "SPEAK" if speak else "SILENT"
        lines.append(f"{m}:{s:04.1f}  {tag:>8}  {dec.get('reason', '')}")
        if speak:
            lines.append(f"{'':>8}  {'':>8}  -> {dec.get('response', '')}")
    lines.append(f"\n{speak_count}/{len(results)} openings -> speak")
    return "\n".join(lines)
