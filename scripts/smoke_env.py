import torch, networkx as nx, importlib, pathlib

print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("NetworkX:", nx.__version__)

# check the mining module only
importlib.import_module("subgraph_mining.decoder")
print("OK import: subgraph_mining.decoder")

# check pretrained checkpoint
ckpt = pathlib.Path("/home/nate/galaxy-mining/neural-subgraph-matcher-miner/ckpt/model.pt")
print("Pretrained model exists:", ckpt.exists(), "|", ckpt)

print("All imports OK.")
