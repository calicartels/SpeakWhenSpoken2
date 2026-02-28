import json
import os

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    key = os.environ.get("SUPERMEMORY_API_KEY", "")
    if not key:
        return None
    from supermemory import Supermemory
    _client = Supermemory(api_key=key)
    return _client


def persist_entity(graph, node_id, meeting_id, participants=None):
    sm = _get_client()
    if sm is None:
        return
    node = graph["nodes"].get(node_id)
    if not node:
        return
    content = json.dumps({
        "label": node["label"], "type": node["type"],
        "mentions": node["mentions"], "meeting_id": meeting_id,
    })
    meta = {"type": node["type"], "meeting_id": meeting_id}
    if participants:
        meta["participants"] = ",".join(str(p) for p in participants)
    sm.documents.add(content=content, container_tag=meeting_id, metadata=meta)


def persist_edge(graph, edge, meeting_id, participants=None):
    sm = _get_client()
    if sm is None:
        return
    src = graph["nodes"].get(edge["source"], {})
    tgt = graph["nodes"].get(edge["target"], {})
    content = json.dumps({
        "source": src.get("label", "?"), "source_type": src.get("type", "?"),
        "target": tgt.get("label", "?"), "target_type": tgt.get("type", "?"),
        "relation": edge["relation"], "meeting_id": meeting_id,
    })
    meta = {"relation": edge["relation"], "meeting_id": meeting_id}
    if participants:
        meta["participants"] = ",".join(str(p) for p in participants)
    sm.documents.add(content=content, container_tag=meeting_id, metadata=meta)


def persist_graph(graph, participants=None):
    mid = graph["meeting_id"]
    for nid in graph["nodes"]:
        persist_entity(graph, nid, mid, participants)
    for edge in graph["edges"]:
        persist_edge(graph, edge, mid, participants)


def search_past(query, limit=10):
    sm = _get_client()
    if sm is None:
        return []
    results = sm.search.memories(q=query, search_mode="hybrid", limit=limit)
    facts = []
    for doc in (results.results or [])[:limit]:
        content = getattr(doc, "memory", None) or getattr(doc, "chunk", None) or ""
        try:
            facts.append(json.loads(content))
        except (json.JSONDecodeError, TypeError):
            facts.append({"raw": str(content)})
    return facts


def render_cross_meeting(facts):
    if not facts:
        return ""
    lines = ["CROSS-MEETING CONTEXT (from past meetings):"]
    for f in facts:
        if "relation" in f:
            lines.append(
                f"  - {f.get('source','?')} --{f['relation']}--> "
                f"{f.get('target','?')} (from {f.get('meeting_id','?')})"
            )
        elif "label" in f:
            lines.append(
                f"  - [{f.get('type','?')}] {f['label']} "
                f"(from {f.get('meeting_id','?')})"
            )
        else:
            lines.append(f"  - {f}")
    return "\n".join(lines)
