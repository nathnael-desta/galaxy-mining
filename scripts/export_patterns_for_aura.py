import json, pickle, pathlib, networkx as nx

# INPUT (your successful fast run)
IN_PKL  = pathlib.Path.home() / "galaxy-mining" / "results" / "mined_patterns_fast.pkl"
# FALLBACK (if you used the full profile)
IN_PKL2 = pathlib.Path.home() / "galaxy-mining" / "results" / "mined_patterns.pkl"

# OUTPUT: a single line you can paste into Aura's Query Editor
OUT_PARAM = pathlib.Path.home() / "galaxy-mining" / "data" / "patterns_param.cypher.txt"

def tool_id_of(node, data):
    # Try to recover the tool id/name from node attributes or the node key itself
    return (
        data.get("id") or
        data.get("label") or
        (node if isinstance(node, str) else str(node))
    )

src = IN_PKL if IN_PKL.exists() else IN_PKL2
if not src.exists():
    raise SystemExit(f"Could not find patterns file: {IN_PKL} or {IN_PKL2}")

pats = pickle.loads(src.read_bytes())
patterns = []

for idx, g in enumerate(pats, 1):
    # pattern id like P001, P002…
    pid = f"P{idx:03d}"
    # Nodes as tool ids/names
    nodes = [tool_id_of(n, g.nodes[n]) for n in g.nodes()]
    # Edges as [{u, v}] with tool ids/names
    edges = []
    for u, v in g.edges():
        uu = tool_id_of(u, g.nodes[u])
        vv = tool_id_of(v, g.nodes[v])
        edges.append({"u": uu, "v": vv})

    patterns.append({
        "pid": pid,
        "size": len(nodes),
        "edge_count": g.number_of_edges(),
        "nodes": nodes,
        "edges": edges,
    })

# Aura requires :param with parentheses around list/map
# We keep keys unquoted (valid in Cypher params) and strings quoted.
# We'll serialize as JSON then massage the keys minimally.
def to_cypher_map(obj):
    # Convert a Python dict/list to a Cypher-like string for :param
    if isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            items.append(f"{k}: {to_cypher_map(v)}")
        return "{" + ", ".join(items) + "}"
    elif isinstance(obj, list):
        return "[" + ", ".join(to_cypher_map(x) for x in obj) + "]"
    elif isinstance(obj, str):
        # escape inner quotes
        s = obj.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{s}'"
    elif obj is None:
        return "null"
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    else:
        return str(obj)

param_line = f":param patterns => ({to_cypher_map(patterns)});"
OUT_PARAM.parent.mkdir(parents=True, exist_ok=True)
OUT_PARAM.write_text(param_line, encoding="utf-8")
print("[WROTE]", OUT_PARAM)
print("Preview:")
print(param_line[:600] + ("..." if len(param_line) > 600 else ""))
