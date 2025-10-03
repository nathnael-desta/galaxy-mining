#!/usr/bin/env python3
import json, random, pathlib, networkx as nx

BASE = pathlib.Path.home() / "galaxy-mining"
DATA = BASE / "data"
DATA.mkdir(parents=True, exist_ok=True)

TOOLS_JSON   = DATA / "tool_index.json"         # already created by your generator
EDGE_LIST    = DATA / "tool_graph.edgelist"     # already created by your generator
SESS_JSON    = DATA / "sessions.json"
LOAD_CYPHER  = DATA / "load_sessions.cypher"

# Tunables
N_SESSIONS   = 2000           # how many sessions to synthesize
LEN_MIN_MAX  = (3, 10)        # min/max tools per session
CHUNK_SIZE   = 300            # sessions per :param chunk in the cypher file (to keep payloads sane)

def load_tools():
    if TOOLS_JSON.exists():
        tools = json.loads(TOOLS_JSON.read_text())
        if isinstance(tools, dict) and "tools" in tools:
            tools = tools["tools"]
        return [str(t) for t in tools]
    raise SystemExit(f"Missing {TOOLS_JSON}. Run your tool generator first.")

def load_graph(tools):
    G = nx.DiGraph()
    if EDGE_LIST.exists():
        # read weighted edgelist u v weight
        G = nx.read_weighted_edgelist(EDGE_LIST, create_using=nx.DiGraph, nodetype=str)
        # Ensure all tools present as nodes (even isolated)
        for t in tools:
            G.add_node(t)
    else:
        # fallback to empty graph with just nodes
        G.add_nodes_from(tools)
    return G

def weighted_next(G, curr):
    """Pick next tool by outgoing edge weights, else random tool."""
    outs = list(G.out_edges(curr, data=True))
    if not outs:
        return random.choice(list(G.nodes))
    # weights default to 1.0 if missing
    neigh = [v for (_, v, _) in outs]
    weights = [float(d.get("weight", 1.0)) for (_, _, d) in outs]
    s = sum(weights)
    if s <= 0:
        return random.choice(neigh)
    r = random.random() * s
    c = 0.0
    for v, w in zip(neigh, weights):
        c += w
        if r <= c:
            return v
    return neigh[-1]

def synth_sessions(tools, G):
    sessions = []
    jid_counter = 1
    for i in range(1, N_SESSIONS + 1):
        sid = f"S{i:04d}"
        user = f"U{random.randint(1, max(100, N_SESSIONS//10)):03d}"
        L = random.randint(*LEN_MIN_MAX)
        seq = []
        curr = random.choice(tools)
        seq.append(curr)
        for _ in range(L - 1):
            curr = weighted_next(G, curr)
            seq.append(curr)
        jobs = []
        for t in seq:
            jid = f"J{jid_counter:07d}"
            jid_counter += 1
            jobs.append({"jid": jid, "tool": t})
        sessions.append({"sid": sid, "user": user, "jobs": jobs})
    return sessions

def write_sessions_json(sessions):
    SESS_JSON.write_text(json.dumps(sessions, indent=2))
    print(f"[WROTE] {SESS_JSON} | sessions={len(sessions)}")

def cypher_literal(obj):
    """Convert Python dict/list to Cypher literal string (not JSON)."""
    if isinstance(obj, dict):
        return "{" + ", ".join(f"{k}: {cypher_literal(v)}" for k, v in obj.items()) + "}"
    elif isinstance(obj, list):
        return "[" + ", ".join(cypher_literal(v) for v in obj) + "]"
    elif isinstance(obj, str):
        return "'" + obj.replace("'", "\\'") + "'"
    else:
        return str(obj)

def write_load_cypher(sessions):
    out = []
    # constraints
    out.append("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tool)    REQUIRE t.id IS UNIQUE;")
    out.append("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Pattern) REQUIRE p.pid IS UNIQUE;")
    out.append("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE;")
    out.append("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User)    REQUIRE u.id IS UNIQUE;")
    out.append("CREATE CONSTRAINT IF NOT EXISTS FOR (j:Job)     REQUIRE j.id IS UNIQUE;")
    out.append("")

    for i in range(0, len(sessions), CHUNK_SIZE):
        chunk = sessions[i:i+CHUNK_SIZE]
        param_name = f"sessions_{i//CHUNK_SIZE}"
        cypher_chunk = cypher_literal(chunk)

        out.append(f":param {param_name} => {cypher_chunk};")
        out.append(f"""
UNWIND ${param_name} AS s
MERGE (sess:Session {{id:s.sid}})
MERGE (u:User {{id:s.user}})
MERGE (u)-[:BELONGS_TO]->(sess)
WITH sess, s
UNWIND s.jobs AS j
MERGE (job:Job {{id:j.jid}})
MERGE (sess)-[:IN_SESSION]->(job)
WITH job, j
MATCH (t:Tool {{id: j.tool}})
MERGE (job)-[:EXECUTED]->(t);
""".strip())
        out.append("")

    LOAD_CYPHER.write_text("\n".join(out))
    print(f"[WROTE] {LOAD_CYPHER}")


def main():
    tools = load_tools()
    G = load_graph(tools)
    sessions = synth_sessions(tools, G)
    write_sessions_json(sessions)
    write_load_cypher(sessions)
    print("[OK] Generated sessions + Aura loader.")

if __name__ == "__main__":
    main()
