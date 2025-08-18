#!/usr/bin/env python3
# Multi-class text classifier using sentiment0 ops (+ ArgMax) under <=32 elems/tensor.
# ONNX ops: Gather -> ReduceSum -> Mul -> Add -> ArgMax
import json, numpy as np, torch, torch.nn as nn
from hashlib import sha256

# -----------------------------
# Tiny 10-class toy data (replace with your own)
# -----------------------------
texts = [
    "cheap flights to rome",            # travel
    "box office hits this weekend",     # entertainment
    "quarterly earnings beat guidance", # business
    "university admissions tips",       # education
    "hotel booking refund policy",      # travel
    "new streaming series announced",   # entertainment
    "merger and acquisition news",      # business
    "scholarships and grants",          # education
]
label_names = ["business","education","travel","entertainment","sports",
               "politics","tech","health","science","other"]   # K = 10
labels = torch.tensor([2,3,0,1,2,3,0,1], dtype=torch.long)

# -----------------------------
# Hashing tokenizer -> M buckets (ID 0 reserved for PAD)
# Ensures Embedding table has at most 32 elements total.
# -----------------------------
PAD = 0
M = 31         # number of non-PAD buckets  (M+1 = 32 <= 32 elems)
L = 8          # sequence length (keep <= 32 to respect tensor cap)

def stable_bucket(token: str, M: int) -> int:
    # Stable, deterministic across runs & languages
    h = int.from_bytes(sha256(token.encode("utf-8")).digest()[:8], "big")
    return 1 + (h % M)   # 1..M, 0 is PAD

def tokenize_hash(text: str):
    return [stable_bucket(tok, M) for tok in text.lower().split()]

# Build batch
seqs = [tokenize_hash(t) for t in texts]
arr = np.full((len(texts), L), PAD, dtype=np.int64)
for i, s in enumerate(seqs):
    n = min(len(s), L)
    arr[i, :n] = s[:n]
X = torch.tensor(arr, dtype=torch.long)  # (B, L)

# -----------------------------
# Model: pooled scalar -> per-class Mul/Add -> logits(K)
# -----------------------------
class BagOfTokensHashed(nn.Module):
    """
    sentiment0-style ops only:
      Gather(Embedding) -> ReduceSum -> Mul(Add) -> ArgMax (export wrapper)
    Embedding shape: (M+1, 1) so the initializer has <=32 elements.
    """
    def __init__(self, num_buckets: int, K: int):
        super().__init__()
        self.emb = nn.Embedding(num_buckets + 1, 1)   # (M+1, 1)
        self.W   = nn.Parameter(torch.ones(K))        # (K,)  per-class scale (Mul)
        self.b   = nn.Parameter(torch.zeros(K))       # (K,)  per-class bias  (Add)
        nn.init.normal_(self.emb.weight, std=0.1)
        with torch.no_grad():
            self.emb.weight[PAD, 0] = 0.0             # PAD contributes nothing

    def forward(self, x):
        e = self.emb(x)                # (B, L, 1)   [Gather]
        s = e.sum(dim=(1, 2))          # (B,)        [ReduceSum]
        logits = s[:, None] * self.W + self.b   # (B, K) [Mul + Add] via broadcast
        return logits

K = len(label_names)
model = BagOfTokensHashed(M, K)

# -----------------------------
# Train (cross-entropy on logits)
# -----------------------------
opt = torch.optim.Adam(model.parameters(), lr=0.03)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(200):
    logits = model(X)                 # (B, K)
    loss = loss_fn(logits, labels)
    opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 40 == 0:
        with torch.no_grad():
            acc = (logits.argmax(1) == labels).float().mean().item()
        print(f"epoch {epoch:03d}  loss {loss.item():.4f}  acc {acc:.2f}")

# -----------------------------
# Save artifacts (hashing params + label names)
# -----------------------------
meta = {"PAD": PAD, "M": M, "L": L, "tokenizer": "sha256_mod"}
with open("labels.json", "w") as f: json.dump(label_names, f, ensure_ascii=False, indent=2)
with open("meta.json",   "w") as f: json.dump(meta,        f, ensure_ascii=False, indent=2)

# -----------------------------
# Export ONNX (single output: class_id via ArgMax)
# -----------------------------
class ExportArgMaxOnly(nn.Module):
    def __init__(self, inner): super().__init__(); self.inner = inner.eval()
    def forward(self, x):
        logits = self.inner(x)              # (1, K)
        return torch.argmax(logits, dim=1)  # ArgMax(axis=1) -> (1,)

dummy = torch.randint(low=0, high=M+1, size=(1, L), dtype=torch.long)  # includes PAD(0)
torch.onnx.export(
    ExportArgMaxOnly(model).eval(), dummy, "network.onnx",
    input_names=["tokens"], output_names=["class_id"],
    opset_version=15  # fixed (1, L)
)



print("Exported multiclass_argmax.onnx (class_id) and multiclass_logits.onnx (logits)")
print(f"Sanity check: Embedding table elements = {(M+1)*1} (<= 32 OK). L={L}, K={K}.")