# Auto-Inference-Optimiser

An autonomous agent that optimises LLM inference speed on Apple Silicon, inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

Point an AI coding agent at this repo and let it run experiments overnight — it hill-climbs on tokens/sec by modifying the inference pipeline, using git commits as experiment tracking.

## How It Works

The core pattern: **lock the evaluation, open the implementation, let an AI agent hill-climb forever.**

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT LOOP (forever)                      │
│                                                             │
│  1. Read current inference.py                               │
│  2. Make an optimisation change                             │
│  3. git commit                                              │
│  4. python prepare.py > run.log 2>&1                        │
│  5. Check: did avg_generation_tps improve?                  │
│  6. Check: does quality_pass == True?                       │
│  7. If both YES → keep. If NO → git reset --hard HEAD~1    │
│  8. Log results to results.tsv                              │
│  9. Go to 1.                                                │
└─────────────────────────────────────────────────────────────┘
```

| File | Role | Agent Can Edit? |
|---|---|---|
| `prepare.py` | Evaluation harness, benchmarks, quality gate | No |
| `inference.py` | The full inference pipeline | **Yes** |
| `program.md` | Agent instructions | No |

## The Optimisation Search Space

The agent can modify anything in `inference.py`:

- **KV cache**: size limits, quantisation (4-bit/8-bit), group size
- **Sampling**: temperature, top-p, top-k, min-p, repetition penalty
- **Prefill**: chunk size, memory management
- **Metal memory**: cache limits, buffer management
- **Architecture**: custom generate loops, speculative decoding, prompt caching
- **Compilation**: `mx.compile` for kernel fusion

## Metrics

| Metric | Direction | Description |
|---|---|---|
| `avg_generation_tps` | Higher is better | Decode tokens per second (primary) |
| `avg_prompt_tps` | Higher is better | Prefill tokens per second |
| `avg_peak_memory_gb` | Lower is better | Peak Metal memory usage |
| `avg_perplexity` | Must stay below 50.0 | Quality gate — prevents speed hacks that destroy output |

## Setup

```bash
pip install -r requirements.txt
```

## Run the Benchmark

```bash
# Run evaluation (downloads model on first run)
python prepare.py

# Check results
cat run.log
```

## Run with an AI Agent

Point any AI coding agent (Claude Code, Codex, Cursor, etc.) at this repo with `program.md` as the system prompt. The agent will:

1. Create a branch
2. Run baseline evaluation
3. Start the optimisation loop
4. Keep going until you stop it

```bash
# Example with Claude Code
claude --system-prompt program.md
```

## Benchmark Model

Uses `mlx-community/Qwen2.5-0.5B-Instruct-4bit` — a 0.5B parameter model in 4-bit quantisation (~350MB). Small enough for any MacBook, large enough to produce meaningful inference patterns.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~2GB free RAM
