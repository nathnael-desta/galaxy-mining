import pickle, pathlib, networkx as nx
out = pathlib.Path.home()/ "galaxy-mining/results/mined_patterns_fast.pkl"
print("exists:", out.exists(), "| size:", out.stat().st_size, "bytes")

if out.exists():
    pats = pickle.loads(out.read_bytes())
    print("patterns found:", len(pats))
    for i, g in enumerate(pats[:3], 1):
        if isinstance(g, (nx.Graph, nx.DiGraph)):
            print(f"[{i}] nodes={g.number_of_nodes()}, edges={g.number_of_edges()}")
            print(" sample nodes:", list(g.nodes())[:6])
            print(" sample edges:", list(g.edges())[:6])