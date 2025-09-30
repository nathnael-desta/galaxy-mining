#!/usr/bin/env python3
import os, json, pickle
from pathlib import Path

from neo4j import GraphDatabase
from dotenv import load_dotenv
import torch
from torch_geometric.data import Data
import networkx as nx

# ---------- config ----------
OUT_DIR = Path.home() / "galaxy-mining" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# env vars (set in your ~/.env earlier)
load_dotenv(Path.home() / "galaxy-mining" / ".env")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    raise SystemExit("NEO4J_URI/USER/PASSWORD not set. Put them in ~/galaxy-mining/.env")

# ---------- fetch from Aura ----------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def fetch_tools(session):
    q = "MATCH (t:Tool) RETURN t.id AS id ORDER BY id"
    return [rec["id"] for rec in session.run(q)]

def fetch_edges(session):
    # weight defaults to 1 if missing; directed edges
    q = """
    MATCH (a:Tool)-[r:TOOL_CO_OCCURRENCE]->(b:Tool)
    RETURN a.id AS u, b.id AS v, coalesce(r.weight, 1) AS w
    """
    for rec in session.run(q):
        yield rec["u"], rec["v"], int(rec["w"])

with driver.session() as session:
    tools = fetch_tools(session)
    edges = list(fetch_edges(session))

driver.close()

print(f"[Aura] Tools: {len(tools)} | Edges: {len(edges)}")

# ---------- build index maps ----------
tool2idx = {t: i for i, t in enumerate(tools)}
idx2tool = {i: t for t, i in tool2idx.items()}

# ---------- build PyG tensors ----------
if edges:
    src = [tool2idx[u] for (u, _, _) in edges]
    dst = [tool2idx[v] for (_, v, _) in edges]
    w   = [int(w) for (_, _, w) in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float)
else:
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_weight = torch.empty((0,), dtype=torch.float)

# simple 1-d node features (can be replaced later)
x = torch.ones((len(tools), 1), dtype=torch.float)

data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
print(data)

# ---------- also build a NetworkX DiGraph ----------
G = nx.DiGraph()
G.add_nodes_from(tools)
for (u, v, w) in edges:
    G.add_edge(u, v, weight=int(w))

# ---------- write outputs ----------
torch.save(data, OUT_DIR / "aura_tool_graph.pt")
(OUT_DIR / "aura_tool_index.json").write_text(json.dumps(
    {"tool2idx": tool2idx, "idx2tool": idx2tool}, indent=2))

with (OUT_DIR / "aura_tool_graph.edgelist").open("w") as f:
    for (u, v, w) in edges:
        f.write(f"{u}\t{v}\t{w}\n")

with (OUT_DIR / "aura_tool_graph.pkl").open("wb") as f:
    pickle.dump(G, f)

print(f"[WROTE] {OUT_DIR/'aura_tool_graph.pt'}")
print(f"[WROTE] {OUT_DIR/'aura_tool_index.json'}")
print(f"[WROTE] {OUT_DIR/'aura_tool_graph.edgelist'}")
print(f"[WROTE] {OUT_DIR/'aura_tool_graph.pkl'}")

# quick sanity
top5 = sorted(edges, key=lambda e: e[2], reverse=True)[:5]
print("[Top edges]", top5)
