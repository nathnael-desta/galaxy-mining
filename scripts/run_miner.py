import os, sys, runpy, pathlib, multiprocessing as mp

def patch_sageconv_edge_weight():
    """Inject a default all-ones edge_weight into common.models.SAGEConv.forward when missing."""
    try:
        import torch
        from common import models
        if not hasattr(models, "SAGEConv"):
            print("[WARN] common.models.SAGEConv not found; skipping patch")
            return
        _orig_forward = models.SAGEConv.forward
        def _forward_with_edge_weight(self, x, edge_index, *args, **kwargs):
            if ("edge_weight" not in kwargs) or (kwargs["edge_weight"] is None):
                E = edge_index.size(1)
                kwargs["edge_weight"] = torch.ones(E, dtype=torch.float, device=edge_index.device)
            return _orig_forward(self, x, edge_index, *args, **kwargs)
        models.SAGEConv.forward = _forward_with_edge_weight
        print("[PATCH] Applied default edge_weight patch to common.models.SAGEConv.forward")
    except Exception as e:
        print(f"[WARN] Could not patch SAGEConv.forward: {e}")

def main():
    # Safer multiprocessing settings for Linux
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    BASE   = pathlib.Path.home() / "galaxy-mining"
    CKPT   = BASE / "neural-subgraph-matcher-miner" / "ckpt" / "model.pt"
    PICKLE = BASE / "data" / "aura_tool_graph_attr_weighted.pkl"
    OUT    = BASE / "results" / "mined_patterns.pkl"
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Apply the edge_weight patch in THIS process (inherited by workers)
    patch_sageconv_edge_weight()

    # Use sampling branch so anchors get created (node_anchored=True by default in your config)
    argv = [
        "subgraph_mining.decoder",
        "--model_path", str(CKPT),
        "--dataset", str(PICKLE),
        "--graph_type", "directed",
        "--sample_method", "tree",
        "--n_neighborhoods", "2000",
        "--subgraph_sample_size", "256",
        "--radius", "2",
        "--batch_size", "16",
        "--search_strategy", "greedy",
        "--min_pattern_size", "3",
        "--max_pattern_size", "8",
        "--out_path", str(OUT),
    ]

    print("Running decoder with args:\n  ", " ".join(argv[1:]))

    _old_argv = sys.argv[:]
    try:
        sys.argv = argv
        runpy.run_module("subgraph_mining.decoder", run_name="__main__")
    finally:
        sys.argv = _old_argv

    print(f"[NOTE] Expected output at: {OUT}")

if __name__ == "__main__":
    main()