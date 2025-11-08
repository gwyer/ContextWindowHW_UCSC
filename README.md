# Context Window Homework - UCSC

This project explores context window limitations and recall performance in LLMs using local Ollama models.

## Overview

The project consists of three main components:

1. **Similarity Probe** (`similarity_probe.py`) - Analyzes cosine similarity between target records and distractors
2. **Recall Experiment** (`run_recall_ollama.py`) - Tests LLM recall performance with varying context window sizes
3. **Visualization** (`plot_recall_curves.py`) - Generates recall curves and performance plots

## Requirements

- Python 3.8+
- Ollama running locally (default: `http://localhost:11434`)
- Dependencies listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Similarity Probe

Generates embeddings and calculates similarity between target and distractor records:

```bash
# High similarity mode (near-duplicates)
python similarity_probe.py 500 high

# Low similarity mode (random strings)
python similarity_probe.py 500 low
```

**Output:**
- `probe_high_similarities.csv` / `probe_low_similarities.csv`
- `probe_high_hist.png` / `probe_low_hist.png`
- `probe_high_scatter.png` / `probe_low_scatter.png`

### 2. Recall Experiment

Tests LLM's ability to recall a target record from varying amounts of distractor data:

```bash
# High similarity distractors
python run_recall_ollama.py

# Low similarity distractors
SIMILARITY=low python run_recall_ollama.py
```

**Output:**
- `all_results.csv` - Contains model performance data with columns:
  - `model` - Model name (e.g., llama3.2)
  - `similarity_mode` - high or low
  - `total_tokens` - Total tokens in prompt
  - `target_distance_tokens` - Distance from target to query
  - `correct` - 1 if recall was correct, 0 otherwise

### 3. Visualization

Plot recall curves from experimental results:

```python
import plot_recall_curves as pr

# Plot all models
pr.plot_recall_curves('all_results.csv')

# Plot specific model
pr.plot_recall_curves('all_results.csv', model='llama3.2')
```

## Results

Experimental results are stored in the `results/` directory:

- Similarity analysis CSV files and plots
- Recall experiment data (`all_results.csv`)

### Key Findings

- **Small contexts (N=100)**: 100% recall accuracy
- **Medium contexts (N=300+)**: 0% recall - hits 4096 token context limit
- Models show sharp performance degradation when approaching context window limits

## Configuration

Environment variables:

- `OLLAMA_HOST` - Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_MODEL` - Model for recall experiment (default: `llama3.2`)
- `EMBED_MODEL` - Model for embeddings (default: `llama3.2`)
- `SIMILARITY` - Similarity mode: `high` or `low` (default: `high`)
- `OUT_CSV` - Output CSV path (default: `all_results.csv`)

## Project Structure

```
.
├── similarity_probe.py      # Embedding similarity analysis
├── run_recall_ollama.py     # Recall experiment runner
├── plot_recall_curves.py    # Visualization utilities
├── requirements.txt         # Python dependencies
├── results/                 # Experimental results (CSV, PNG)
└── README.md               # This file
```

## License

Educational project for UCSC coursework.
