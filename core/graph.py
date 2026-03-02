import time
import uuid
import logging

log = logging.getLogger("graph")


RELATIONAL_LABELS = {
    "person owns deadline": ("person", "owns", "deadline"),
    "decision about topic": ("decision", "about", "topic"),
    "commitment by person": ("commitment", "by", "person"),
}


def new_graph(meeting_id=None):
    ts = time.strftime("%Y%m%d-%H%M%S")
    mid = meeting_id or f"meeting-{ts}-{uuid.uuid4().hex[:6]}"
    return {
        "meeting_id": mid,
        "nodes": {},
        "edges": [],
        "version": 0,
    }


def node_id(label, node_type):
    return f"{node_type}:{label.lower().strip()}"


def add_entity(graph, label, node_type, timestamp):
    nid = node_id(label, node_type)
    if nid in graph["nodes"]:
        graph["nodes"][nid]["mentions"] += 1
        graph["nodes"][nid]["last_seen"] = timestamp
    else:
        graph["nodes"][nid] = {
            "id": nid, "label": label, "type": node_type,
            "mentions": 1, "first_seen": timestamp, "last_seen": timestamp,
        }
    graph["version"] += 1
    return nid


def add_relation(graph, src_label, src_type, tgt_label, tgt_type, relation, ts):
    src = add_entity(graph, src_label, src_type, ts)
    tgt = add_entity(graph, tgt_label, tgt_type, ts)
    for e in graph["edges"]:
        if e["source"] == src and e["target"] == tgt and e["relation"] == relation:
            return
    graph["edges"].append({
        "source": src, "target": tgt, "relation": relation,
        "timestamp": ts, "meeting_id": graph["meeting_id"],
    })
    graph["version"] += 1
    log.info(f"New Edge: {src_label} --{relation}--> {tgt_label}")


def add_relational_edge(graph, label, text, ts):
    src_type, rel, tgt_type = RELATIONAL_LABELS[label]
    parts = text.split()
    split_idx = None
    for i, w in enumerate(parts):
        if w.lower() == rel:
            split_idx = i
            break
    if split_idx and 0 < split_idx < len(parts) - 1:
        src_text = " ".join(parts[:split_idx])
        tgt_text = " ".join(parts[split_idx + 1:])
        add_relation(graph, src_text, src_type, tgt_text, tgt_type, rel, ts)
    else:
        add_entity(graph, text, label, ts)


def ingest_gliner(graph, entities, ts):
    if not entities:
        return
    log.debug(f"Ingesting {len(entities)} GLiNER entities")
    simple = []
    for ent in entities:
        if ent["label"].lower() in RELATIONAL_LABELS:
            add_relational_edge(graph, ent["label"].lower(), ent["text"], ts)
        else:
            add_entity(graph, ent["text"], ent["label"], ts)
            simple.append(ent)

    if len(simple) >= 2:
        for i in range(len(simple)):
            for j in range(i + 1, len(simple)):
                a, b = simple[i], simple[j]
                if a["label"] != b["label"]:
                    add_relation(graph, a["text"], a["label"],
                                 b["text"], b["label"], "co-mentioned", ts)


def serialize(graph):
    return {
        "meeting_id": graph["meeting_id"],
        "version": graph["version"],
        "nodes": list(graph["nodes"].values()),
        "edges": graph["edges"],
    }


def render_for_llm(graph):
    if not graph["nodes"]:
        return "(no graph data yet)"
    lines = []
    for n in graph["nodes"].values():
        lines.append(f"- [{n['type']}] {n['label']} (x{n['mentions']})")
    for e in graph["edges"]:
        src = graph["nodes"].get(e["source"], {}).get("label", "?")
        tgt = graph["nodes"].get(e["target"], {}).get("label", "?")
        lines.append(f"  {src} --{e['relation']}--> {tgt}")
    return "\n".join(lines)
