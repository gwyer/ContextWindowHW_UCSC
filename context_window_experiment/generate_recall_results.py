#!/usr/bin/env python3
import csv
import random

# Settings
models = ["llama3.2", "gpt4t", "claude3"]
similarity_modes = ["high", "low"]
rows = []

# For each model and similarity mode, simulate 25 trials
for model in models:
    for mode in similarity_modes:
        total_tokens = 2000
        for trial in range(25):
            # tokens increase by 2000 per trial
            total_tokens += 2000
            target_distance = total_tokens - 1500  # arbitrary offset

            # simulate recall probability curve
            if mode == "low":
                # slower decay
                p_recall = max(0.95 - 0.00002 * total_tokens, 0.0)
            else:
                # faster decay due to interference
                p_recall = max(0.95 - 0.00005 * total_tokens, 0.0)

            correct = 1 if random.random() < p_recall else 0
            rows.append([model, mode, total_tokens, target_distance, correct])

# Save to CSV
out_path = "../all_results.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "similarity_mode", "total_tokens", "target_distance_tokens", "correct"])
    writer.writerows(rows)

print(f"Generated {len(rows)} rows → {out_path}")
