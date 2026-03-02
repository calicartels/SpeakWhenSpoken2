import json
import os
import logging

log = logging.getLogger("supermemory")
sm_client = None


def get_client():
    global sm_client
    if sm_client is not None:
        return sm_client
    key = os.environ.get("SUPERMEMORY_API_KEY", "")
    if not key:
        return None
    from supermemory import Supermemory
    sm_client = Supermemory(api_key=key)
    return sm_client


def persist_entity(graph, node_id, meeting_id, participants=None):
    sm = get_client()
    if sm is None:
        return
    node = graph["nodes"].get(node_id)
    if not node:
        return

    label = node["label"]
    ntype = node["type"]
    content = f"[Meeting {meeting_id}] Participants discussed {label}, which is a {ntype}."

    meta = {"type": ntype, "meeting_id": meeting_id, "is_entity": "true"}
    if participants:
        meta["participants"] = ",".join(str(p) for p in participants)

    sm.documents.add(content=content, container_tag=meeting_id, metadata=meta)


def persist_edge(graph, edge, meeting_id, participants=None):
    sm = get_client()
    if sm is None:
        return
    src = graph["nodes"].get(edge["source"], {})
    tgt = graph["nodes"].get(edge["target"], {})

    src_label = src.get("label", "Unknown")
    tgt_label = tgt.get("label", "Unknown")
    relation = edge["relation"]
    content = f"[Meeting {meeting_id}] {src_label} {relation} {tgt_label}."

    meta = {"relation": relation, "meeting_id": meeting_id, "is_edge": "true"}
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
    sm = get_client()
    if sm is None:
        return []
    results = sm.search.memories(q=query, search_mode="hybrid", limit=limit)
    facts = []
    for doc in (results.results or [])[:limit]:
        content = getattr(doc, "memory", None) or getattr(doc, "chunk", None) or ""
        facts.append({"raw": str(content)})
    return facts


def render_cross_meeting(facts):
    if not facts:
        return ""
    lines = ["CROSS-MEETING CONTEXT (from past meetings):"]
    for f in facts:
        lines.append(f"  - {f.get('raw', str(f))}")
    return "\n".join(lines)
