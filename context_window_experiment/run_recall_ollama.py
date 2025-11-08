#!/usr/bin/env python3
# Real recall experiment using a local Ollama model.
# Writes: all_results.csv (append-safe)

import os, csv, re, time, json, random, string, requests
from pathlib import Path

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL       = os.environ.get("OLLAMA_MODEL", "llama3.2")   # your llama3.2
TEMP        = float(os.environ.get("TEMP", "0.0"))         # deterministic
OUT_CSV     = os.environ.get("OUT_CSV", "../all_results.csv")

# ---------------- helpers ----------------
def call_generate(prompt: str) -> dict:
    """
    Use /api/generate (non-streaming) so we receive token counts.
    Returns dict with keys: response, prompt_eval_count, eval_count
    """
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMP,
            # keep ctx the model default; you can set num_ctx if desired
        }
    }
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()

def gen_high_similarity(idx: int):
    # near-duplicates to induce interference
    name = f"Alpha_{idx:05d}"
    code = f"ZK32F{idx:05d}"
    return name, code

def gen_low_similarity(idx: int):
    name = "n_" + ''.join(random.choices(string.ascii_lowercase, k=8)) + f"_{idx}"
    code = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    return name, code

def record_line(name: str, code: str) -> str:
    # one-line, easy to parse later
    return f"Record(name={name}, code={code})"

def build_prompt(target_name, target_code, distractors, filler_every=10):
    """
    Serialize a 'chat-like' prompt for /api/generate:
    We’ll label turns with 'User:' / 'Assistant:' to mimic a dialogue.
    """
    lines = []
    lines.append("System: You are a helpful assistant. Answer concisely.\n")
    lines.append(f"User: I will give you a target record once. Remember it exactly.\n")
    lines.append(f"User: TARGET --> {record_line(target_name, target_code)}\n")
    lines.append("Assistant: Noted.\n")

    # add some unrelated chatter and the distractors
    for i, (name, code) in enumerate(distractors):
        if i % filler_every == 0:
            lines.append(f"User: Brief aside #{i//filler_every}: the weather is fine.\n")
            lines.append("Assistant: Okay.\n")
        lines.append(f"User: {record_line(name, code)}\n")
        # (no assistant echo to keep size bounded)

    # final query
    lines.append("User: What was the TARGET record (name and code) I gave earlier?\n"
                 "Assistant:")
    return "".join(lines)

def parse_answer(text: str):
    """
    Try to extract name and code from the model's reply.
    Handles formats like:
      - Record(name=Alpha42, code=ZK32F9)
      - name = Alpha42, code = ZK32F9
      - Alpha42 / ZK32F9
    """
    # Try strict 'Record(...)' first
    m = re.search(r"Record\s*\(\s*name\s*=\s*([A-Za-z0-9_]+)\s*,\s*code\s*=\s*([A-Za-z0-9]+)\s*\)", text)
    if m:
        return m.group(1), m.group(2)

    # name=..., code=...
    m = re.search(r"name\s*=\s*([A-Za-z0-9_]+).{0,40}?code\s*=\s*([A-Za-z0-9]+)", text, re.I|re.S)
    if m:
        return m.group(1), m.group(2)

    # fallback: two tokens separated by punctuation or slash
    m = re.search(r"\b([A-Za-z0-9_]{4,})\b[^A-Za-z0-9_]+([A-Za-z0-9]{4,})\b", text)
    if m:
        return m.group(1), m.group(2)

    return None, None

def ensure_csv_header(path):
    if not Path(path).exists():
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model","similarity_mode","total_tokens","target_distance_tokens","correct"])

# crude token estimator (for distance only), in case you want both
def est_tokens_from_chars(s: str) -> int:
    # heuristic: ~4 chars per token (English-ish)
    return max(1, round(len(s)/4))

# ---------------- experiment ----------------
def run_real_trials(
    similarity_mode="high",
    trials=20,
    distractor_counts=(100, 300, 600, 1000),
    seed=42,
):
    random.seed(seed)
    ensure_csv_header(OUT_CSV)

    target_name, target_code = "Alpha42", "ZK32F9"

    for N in distractor_counts:
        for t in range(trials):
            # make distractors
            distractors = []
            for i in range(N):
                if similarity_mode == "high":
                    distractors.append(gen_high_similarity(i))
                else:
                    distractors.append(gen_low_similarity(i))

            # build prompt
            prompt = build_prompt(target_name, target_code, distractors)

            # For distance estimate: count chars after the TARGET line
            # (Find index of the TARGET marker and measure tail)
            tgt_marker = f"TARGET --> {record_line(target_name, target_code)}"
            pos = prompt.find(tgt_marker)
            tail_chars = len(prompt) - (pos + len(tgt_marker)) if pos >= 0 else len(prompt)
            target_distance_tokens = max(1, round(tail_chars / 4))

            # call model
            resp = call_generate(prompt)
            answer = resp.get("response","")
            total_tokens = resp.get("prompt_eval_count") or est_tokens_from_chars(prompt)

            # parse + grade
            got_name, got_code = parse_answer(answer)
            correct = int((got_name == target_name) and (got_code == target_code))

            # append row
            with open(OUT_CSV, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    MODEL,
                    similarity_mode,
                    total_tokens,
                    target_distance_tokens,
                    correct
                ])

            print(f"[{MODEL}][{similarity_mode}] N={N:4d} trial={t+1:02d} "
                  f"tokens={total_tokens} dist≈{target_distance_tokens} "
                  f"correct={correct}")
            # be nice to your CPU/GPU
            time.sleep(0.05)

if __name__ == "__main__":
    # Examples:
    #   python run_recall_ollama.py                 # high-sim trials
    #   SIMILARITY=low python run_recall_ollama.py  # low-sim trials
    sim = os.environ.get("SIMILARITY", "high")
    run_real_trials(similarity_mode=sim)
