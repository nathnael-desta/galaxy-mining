#!/usr/bin/env python3
import os, json, random, math
from pathlib import Path
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

# ---------- configuration (tweak safely) ----------
SEED = 42
NUM_TOOLS = 120           # total distinct tools
NUM_USERS = 80            # synthetic users
SESSIONS_PER_USER = (15, 45)   # min/max sessions per user
SESSION_LEN = (5, 12)     # min/max tools per session
MOTIF_INSTANCES = 600     # approximate number of motif embeddings overall
OUT_DIR = Path.home() / "galaxy-mining" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# named tools (seed realism); the rest will be auto-filled
BASE_TOOLS = [
    "FastQC","Trimmomatic","BWA-MEM","STAR","HISAT2","Bowtie2","Samtools_Sort",
    "Samtools_Index","Picard_MarkDuplicates","GATK_HaplotypeCaller","Bedtools_intersect",
    "Bedtools_merge","Cutadapt","featureCounts","HTSeq_Count","MultiQC","FreeBayes",
    "VCFtools","DESeq2","EdgeR","Kallisto","Salmon","BCFtools","MACS2","IGV",
    "Qualimap","StringTie","Subread_align","SPAdes","Velvet","MUSCLE","MAFFT",
    "BLASTn","BLASTp","Diamond","Minimap2","NanoPlot","QUAST","Kraken2","Bracken",
    "Python_script","Rscript","Jupyter_Notebook","awk_filter","grep_filter"
]
# recurrent motifs we want the miner to rediscover
MOTIFS = [
    ("FastQC","Trimmomatic","BWA-MEM","Samtools_Sort","Samtools_Index","Picard_MarkDuplicates"),
    ("FastQC","Cutadapt","STAR","featureCounts","MultiQC"),
    ("Minimap2","Samtools_Sort","Samtools_Index","Bedtools_intersect"),
    ("Kallisto","DESeq2","MultiQC"),
    ("Salmon","EdgeR","MultiQC"),
    ("BWA-MEM","GATK_HaplotypeCaller","VCFtools"),
    ("BLASTn","Bedtools_intersect","Python_script"),
]

# probs controlling sequence generation
P_CONTINUE_MOTIF = 0.75   # while inside motif, keep following it
P_JUMP_MOTIF = 0.25       # chance to inject/start a motif in a session
P_BACKEDGE = 0.05         # add occasional A->A or A->previous transitions
# --------------------------------------------------

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# build full tool list
tools = list(BASE_TOOLS)
if len(tools) < NUM_TOOLS:
    extra = [f"Tool_{i:03d}" for i in range(1, NUM_TOOLS - len(tools) + 1)]
    tools.extend(extra)
tools = tools[:NUM_TOOLS]
tool_set = set(tools)

# helper: generate one session sequence
def gen_session():
    L = random.randint(*SESSION_LEN)
    seq = []
    i = 0
    while i < L:
        # chance to start a known motif (if it fits)
        if random.random() < P_JUMP_MOTIF:
            motif = random.choice(MOTIFS)
            # ensure motif tools exist in our tool universe
            motif = tuple(t for t in motif if t in tool_set)
            if len(motif) >= 2:
                # place part/all of the motif
                take = min(len(motif), L - i)
                seq.extend(motif[:take])
                i += take
                continue
        # otherwise pick a random tool (avoid immediate duplicate)
        t = random.choice(tools)
        if seq and t == seq[-1] and random.random() > 0.2:
            continue
        seq.append(t); i += 1
    # sprinkle occasional backedges by duplicating a recent tool
    if len(seq) >= 3 and random.random() < P_BACKEDGE:
        pos = random.randint(1, len(seq)-2)
        seq.insert(pos+1, seq[pos])
    return seq

# generate sessions per user, plus guarantee some motif instances
sessions = []
# hard-embed extra motif instances to boost frequency
for _ in range(MOTIF_INSTANCES):
    m = random.choice(MOTIFS)
    m = tuple(t for t in m if t in tool_set)
    if len(m) >= 3:
        # maybe pad with random head/tail
        head = [random.choice(tools)] if random.random() < 0.3 else []
        tail = [random.choice(tools)] if random.random() < 0.3 else []
        sessions.append(head + list(m) + tail)

for _ in range(NUM_USERS):
    k = random.randint(*SESSIONS_PER_USER)
    for __ in range(k):
        sessions.append(gen_session())

random.shuffle(sessions)

# build directed co-occurrence counts from session sequences
pair_counts = {}
for seq in sessions:
    for a, b in zip(seq, seq[1:]):
        if a == b:  # allow self if it happens
            pass
        pair_counts[(a,b)] = pair_counts.get((a,b), 0) + 1

# make a directed graph with weights
G = nx.DiGraph()
G.add_nodes_from(tools)
for (a,b), w in pair_counts.items():
    if a in tool_set and b in tool_set:
        if G.has_edge(a,b):
            G[a][b]['weight'] += w
        else:
            G.add_edge(a,b, weight=w)

# summary
num_edges = G.number_of_edges()
num_nodes = G.number_of_nodes()
print(f"[OK] Tools: {num_nodes} | Co-occurrence edges: {num_edges} | Sessions: {len(sessions)}")

# --------- write outputs ----------
OUT_DIR.mkdir(exist_ok=True, parents=True)

import pickle

# 1) NetworkX graph (pickle for inspection / later conversion)
nx_path = OUT_DIR / "tool_graph.pkl"
with nx_path.open("wb") as f:
    pickle.dump(G, f)


# 2) Simple edgelist with weights (tab-separated)
edgelist_path = OUT_DIR / "tool_graph.edgelist"
with edgelist_path.open("w") as f:
    for u,v,data in G.edges(data=True):
        f.write(f"{u}\t{v}\t{int(data.get('weight',1))}\n")

# 3) Mapping (tool -> index) for PyG
tool2idx = {t:i for i,t in enumerate(sorted(G.nodes()))}
idx2tool = {i:t for t,i in tool2idx.items()}
with (OUT_DIR / "tool_index.json").open("w") as f:
    json.dump({"tool2idx": tool2idx, "idx2tool": idx2tool}, f, indent=2)

# 4) PyG Data (x, edge_index, edge_weight)
edges = [(tool2idx[u], tool2idx[v], int(d.get('weight',1))) for u,v,d in G.edges(data=True)]
if edges:
    src, dst, w = zip(*edges)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float)
else:
    edge_index = torch.empty((2,0), dtype=torch.long)
    edge_weight = torch.empty((0,), dtype=torch.float)

x = torch.ones((len(tool2idx), 1), dtype=torch.float)  # 1-dim node features
data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
torch.save(data, OUT_DIR / "tool_graph.pt")

# 5) Cypher for Tools
cy_tools = OUT_DIR / "load_tools.cypher"
with cy_tools.open("w") as f:
    f.write("// Create Tool nodes\n")
    for t in sorted(G.nodes()):
        safe = t.replace("'", "\\'")
        f.write(f"MERGE (:Tool {{id: '{safe}'}});\n")

# 6) Cypher for TOOL_CO_OCCURRENCE (weights as frequency)
cy_edges = OUT_DIR / "load_tool_cooc.cypher"
with cy_edges.open("w") as f:
    f.write("// Create TOOL_CO_OCCURRENCE relationships with weight\n")
    for u, v, d in G.edges(data=True):
        w = int(d.get('weight',1))
        u_safe = u.replace("'", "\\'")
        v_safe = v.replace("'", "\\'")
        f.write(
            "MATCH (a:Tool {id:'%s'}), (b:Tool {id:'%s'}) "
            "MERGE (a)-[r:TOOL_CO_OCCURRENCE]->(b) "
            "ON CREATE SET r.weight=%d "
            "ON MATCH SET r.weight=%d;\n" % (u_safe, v_safe, w, w)
        )

print(f"[WROTE] {nx_path}")
print(f"[WROTE] {edgelist_path}")
print(f"[WROTE] {OUT_DIR / 'tool_index.json'}")
print(f"[WROTE] {OUT_DIR / 'tool_graph.pt'}")
print(f"[WROTE] {cy_tools}")
print(f"[WROTE] {cy_edges}")
