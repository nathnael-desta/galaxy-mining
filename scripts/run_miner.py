#!/usr/bin/env python3
import os, sys, runpy, pathlib, multiprocessing as mp

def patch_sageconv_edge_weight():
    """Inject default ones edge_weight into common.models.SAGEConv.forward when missing."""
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

def patch_numpy_visible_deprecation():
    """Provide np.VisibleDeprecationWarning on NumPy 2.x where it's missing."""
    try:
        import numpy as np
        if not hasattr(np, "VisibleDeprecationWarning"):
            try:
                # Some NumPy builds expose it here
                from numpy.exceptions import VisibleDeprecationWarning as VDW
            except Exception:
                class VDW(Warning):
                    pass
            np.VisibleDeprecationWarning = VDW
            print("[PATCH] Added numpy.VisibleDeprecationWarning alias")
    except Exception as e:
        print(f"[WARN] Could not patch NumPy VisibleDeprecationWarning: {e}")

def build_argv(profile="FAST"):
    BASE   = pathlib.Path.home() / "galaxy-mining"
    CKPT   = BASE / "neural-subgraph-matcher-miner" / "ckpt" / "model.pt"
    PICKLE = BASE / "data" / "aura_tool_graph_attr_weighted.pkl"
    OUT    = BASE / "results" / ("mined_patterns_fast.pkl" if profile=="FAST" else "mined_patterns.pkl")

    common = [
        "subgraph_mining.decoder",
        "--model_path", str(CKPT),
        "--dataset", str(PICKLE),
        "--graph_type", "directed",
        "--search_strategy", "greedy",
        "--min_pattern_size", "3",
        "--max_pattern_size", "8",
        "--out_path", str(OUT),
    ]

    if profile == "FAST":
        # Much faster dev run (few minutes or less on CPU)
        common += [
            "--sample_method", "tree",
            "--n_neighborhoods", "400",    # was 2000
            "--subgraph_sample_size", "128",# was 256
            "--radius", "2",
            "--batch_size", "32",           # was 16
            "--n_trials", "150",            # was 1000 (default)
            "--out_batch_size", "5"         # fewer motifs per size
        ]
    else:
        # Original-ish heavier run (what you did)
        common += [
            "--sample_method", "tree",
            "--n_neighborhoods", "2000",
            "--subgraph_sample_size", "256",
            "--radius", "2",
            "--batch_size", "16",
            "--n_trials", "1000",
            "--out_batch_size", "10"
        ]
    return common

def main():
    # Safer multiprocessing
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Patches in the parent so workers inherit
    patch_numpy_visible_deprecation()
    patch_sageconv_edge_weight()

    profile = os.environ.get("MINER_PROFILE", "FAST").upper()  # FAST or FULL
    argv = build_argv(profile)

    print("Running decoder with args:\n  ", " ".join(argv[1:]))

    _old_argv = sys.argv[:]
    try:
        sys.argv = argv
        runpy.run_module("subgraph_mining.decoder", run_name="__main__")
    finally:
        sys.argv = _old_argv

if __name__ == "__main__":
    main()