import pickle, pathlib, networkx as nx

base = pathlib.Path.home()/ "galaxy-mining" / "data"
src  = base / "aura_tool_graph_attr.pkl"          # from previous step
dst  = base / "aura_tool_graph_attr_weighted.pkl" # new file with edge_weight

G = pickle.loads(src.read_bytes())
if not isinstance(G, (nx.Graph, nx.DiGraph)):
    raise SystemExit(f"Expected NetworkX graph in {src}, got {type(G)}")

# Ensure directed (your TOOL_CO_OCCURRENCE is directed)
if not G.is_directed():
    G = G.to_directed()

# Guarantee both 'weight' and 'edge_weight' on every edge
missing = 0
for u, v, data in G.edges(data=True):
    w = data.get("weight", 1)
    data["weight"] = w
    data["edge_weight"] = float(w)  # tensor will be float
    if "weight" not in data:
        missing += 1

# (Optional) ensure nodes still have readable attrs
for n in G.nodes():
    G.nodes[n].setdefault("label", str(n))
    G.nodes[n].setdefault("id", str(n))

dst.write_bytes(pickle.dumps(G, protocol=pickle.HIGHEST_PROTOCOL))
print(f"[OK] wrote {dst} | nodes={G.number_of_nodes()} edges={G.number_of_edges()}")