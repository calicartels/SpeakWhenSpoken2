import os
import logging
import httpx
import config

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger("decide")
_client = httpx.Client(timeout=10.0)
_api_key = os.environ.get("MERCURY_API_KEY", "")

SYSTEM_DRAFT = (
    "You are an AI meeting participant named Fauna. You are always listening.\n\n"
    "STRICT RULES:\n"
    "- You may ONLY reference information from the TRANSCRIPT below.\n"
    "- You may ONLY reference entities/facts from the MEMORY section.\n"
    "- You may reference CROSS-MEETING CONTEXT if directly relevant.\n"
    "- Do NOT hallucinate names, dates, numbers, or commitments.\n"
    "- If you lack context to say something useful, respond exactly: SILENT\n\n"
    "Draft what you WOULD say if given the floor. 1-2 sentences max.\n\n"
    "Speech act guidance:\n"
    "- DIRECTIVE (question/request): answer it.\n"
    "- CONSTATIVE (statement/opinion): only add if you have relevant info.\n"
    "- ACKNOWLEDGMENT: prefer SILENT.\n"
    "- COMMISSIVE (commitment/plan): note it, don't repeat."
)

SYSTEM_DIRECT = (
    "You are Fauna, an AI meeting participant who was just directly addressed.\n\n"
    "STRICT RULES:\n"
    "- Only reference info from TRANSCRIPT, MEMORY, and CROSS-MEETING CONTEXT.\n"
    "- Do NOT hallucinate. Be concise: 1-3 sentences max.\n"
    "- Respond naturally as a helpful meeting participant."
)


def _classify_speech_act(text):
    lower = text.lower().strip()
    if any(q in lower for q in ["?", "what do you", "can you", "could you",
                                 "do you think", "how about", "should we"]):
        return "DIRECTIVE"
    if any(a in lower for a in ["yeah", "right", "exactly", "sure", "okay",
                                 "i agree", "makes sense", "got it"]):
        return "ACKNOWLEDGMENT"
    if any(c in lower for c in ["i will", "i'll", "let me", "i can",
                                 "we should", "let's", "i promise"]):
        return "COMMISSIVE"
    return "CONSTATIVE"


def build_context(transcript, memory_text, meeting_state, cross_meeting_text=""):
    recent = transcript[-20:] if len(transcript) > 20 else transcript
    lines = []
    for seg in recent:
        speaker = seg.get("speaker", "UNKNOWN")
        lines.append(f"[{speaker}]: {seg.get('text', '')}")
    transcript_block = "\n".join(lines) if lines else "(no transcript yet)"

    last_text = recent[-1].get("text", "") if recent else ""
    speech_act = _classify_speech_act(last_text)
    last_speaker = recent[-1].get("speaker", "UNKNOWN") if recent else "UNKNOWN"

    ctx = (
        f"TRANSCRIPT (recent):\n{transcript_block}\n\n"
        f"LAST UTTERANCE SPEECH ACT: {speech_act} (by {last_speaker})\n\n"
        f"MEMORY (current meeting):\n{memory_text or '(none)'}\n\n"
    )
    if cross_meeting_text:
        ctx += f"{cross_meeting_text}\n\n"
    ctx += f"MEETING STATE:\n{meeting_state or '(initializing)'}"
    return ctx


def draft(transcript, memory_text, meeting_state, cross_meeting_text=""):
    context = build_context(transcript, memory_text, meeting_state, cross_meeting_text)
    return _call_mercury(SYSTEM_DRAFT, context)


def respond_direct(transcript, memory_text, meeting_state, cross_meeting_text=""):
    context = build_context(transcript, memory_text, meeting_state, cross_meeting_text)
    resp = _call_mercury(SYSTEM_DIRECT, context)
    if resp == "SILENT":
        return "I'm here -- could you repeat that?"
    return resp


def _call_mercury(system, user_content):
    if not _api_key:
        log.warning("MERCURY_API_KEY not set — returning SILENT")
        return "SILENT"
    try:
        resp = _client.post(
            config.MERCURY_API_URL,
            headers={"Authorization": f"Bearer {_api_key}", "Content-Type": "application/json"},
            json={
                "model": config.MERCURY_MODEL,
                "max_tokens": 150,
                "temperature": 0.7,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.error(f"Mercury call failed: {e}")
        return "SILENT"
