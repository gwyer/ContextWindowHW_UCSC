#!/usr/bin/env python3
import os
import sys
import json
import time
import math
import random
import string
import subprocess
from typing import List, Tuple
import numpy as np

# Optional: install if missing
# pip install matplotlib pandas requests

import requests
import pandas as pd
import matplotlib.pyplot as plt

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("EMBED_MODEL", "llama3.2")   # change if you prefer another embedding model
USE_HTTP = True                                     # set False to use CLI fallback

# ---------------- Cosine similarity ----------------
def cosine_similarity(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom != 0 else 0.0

# ---------------- Ollama embedding helpers ----------------
def embed_http(text: str) -> List[float]:
    """Ollama HTTP API: POST /api/embeddings {model, prompt} -> {embedding: [...] }"""
    url = f"{OLLAMA_HOST}/api/embeddings"
    resp = requests.post(url, json={"model": MODEL, "prompt": text}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # Some builds return {"embedding": [...]}, others {"embeddings":[[...]]}
    if "embedding" in data and isinstance(data["embedding"], list):
        return data["embedding"]
    if "embeddings" in data and isinstance(data["embeddings"], list):
        return data["embeddings"][0]
    raise ValueError(f"Unexpected embeddings payload: {data.keys()}")

def embed_cli(text: str) -> List[float]:
    """CLI fallback: ollama embed --model <MODEL> "text" -> JSON with embeddings"""
    cmd = ["ollama", "embed", "--model", MODEL, text]
    out = subprocess.check_output(cmd)
    data = json.loads(out.decode("utf-8"))
    # CLI returns {"embeddings":[[...]]}
    if "embeddings" in data and isinstance(data["embeddings"], list):
        return data["embeddings"][0]
    # Some versions: {"embedding":[...]}
    if "embedding" in data:
        return data["embedding"]
    raise ValueError(f"Unexpected CLI embeddings payload: {data.keys()}")

def embed(text: str) -> List[float]:
    if USE_HTTP:
        return embed_http(text)
    return embed_cli(text)

# ---------------- Synthetic data generators ----------------
def gen_high_similarity_record(prefix: str, idx: int) -> Tuple[str, str]:
    # Similar strings (intended to induce interference)
    name = f"{prefix}_{idx:05d}"
    code = f"ZK32F{idx:05d}"
    return name, code

def gen_low_similarity_record(idx: int) -> Tuple[str, str]:
    # Random strings (lower interference)
    name = "n_" + ''.join(random.choices(string.ascii_lowercase, k=8)) + f"_{idx}"
    code = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    return name, code

def as_text(name: str, code: str) -> str:
    # Consistent format for embedding
    return f"Record(name={name}, code={code})"

# ---------------- Main experiment ----------------
def run_probe(n=500, mode="high", seed=42, out_prefix="probe"):
    random.seed(seed)

    # Target
    target_name, target_code = "Alpha42", "ZK32F9"
    target_text = as_text(target_name, target_code)

    print(f"Embedding target with model={MODEL} …")
    t0 = time.time()
    target_vec = embed(target_text)
    print(f"Target embedded in {time.time()-t0:.2f}s; dim={len(target_vec)}")

    # Distractors
    rows = []
    print(f"Generating and embedding {n} distractors (mode={mode}) …")
    for i in range(n):
        if mode == "high":
            name, code = gen_high_similarity_record("Alpha", i)   # near-duplicates
        else:
            name, code = gen_low_similarity_record(i)             # random-ish

        text = as_text(name, code)
        vec = embed(text)
        sim = cosine_similarity(target_vec, vec)
        rows.append({
            "idx": i,
            "name": name,
            "code": code,
            "text": text,
            "cosine_to_target": sim
        })

        if (i+1) % 50 == 0:
            print(f"  … {i+1}/{n} done")

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = f"{out_prefix}_{mode}_similarities.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df)} rows)")

    # Basic stats
    print(df["cosine_to_target"].describe())

    # Plot histogram
    plt.figure()
    plt.hist(df["cosine_to_target"], bins=40)
    plt.title(f"Cosine similarity to target ({mode} similarity, n={n})")
    plt.xlabel("cosine(target, distractor)")
    plt.ylabel("count")
    plt.tight_layout()
    hist_path = f"{out_prefix}_{mode}_hist.png"
    plt.savefig(hist_path, dpi=150)
    print(f"Saved: {hist_path}")

    # Scatter (index vs similarity) to see drift over time/order
    plt.figure()
    plt.scatter(df["idx"], df["cosine_to_target"], s=10)
    plt.title(f"Similarity vs. index ({mode} similarity, n={n})")
    plt.xlabel("distractor index (injection order)")
    plt.ylabel("cosine(target, distractor)")
    plt.tight_layout()
    scatter_path = f"{out_prefix}_{mode}_scatter.png"
    plt.savefig(scatter_path, dpi=150)
    print(f"Saved: {scatter_path}")

if __name__ == "__main__":
    # CLI usage:
    #   python similarity_probe.py              # default: n=500, mode=high
    #   EMBED_MODEL=nomic-embed-text python similarity_probe.py
    #   python similarity_probe.py 1000 low
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    mode = sys.argv[2] if len(sys.argv) > 2 else "high"  # "high" or "low"
    run_probe(n=n, mode=mode)
