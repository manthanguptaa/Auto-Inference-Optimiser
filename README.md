# Auto-Inference-Optimiser

An autonomous agent that optimises LLM inference speed on Apple Silicon, inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

Point an AI coding agent at this repo and let it run experiments overnight — it hill-climbs on tokens/sec by modifying the inference pipeline, using git commits as experiment tracking.

## Results: Claude Opus 4.6 Optimisation Run

**Hardware:** MacBook Pro M4, 48GB RAM
**Model:** `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (0.5B params, 4-bit)
**Agent:** Claude Opus 4.6 via Claude Code
**Experiments:** 16 total, 5 kept, 11 reverted

### Summary

| Metric | Baseline | Optimised | Change |
|---|---|---|---|
| `avg_generation_tps` | 397.0 | 434.9 | **+9.5%** |
| `avg_prompt_tps` | 2,467.7 | 3,180.0 | **+28.9%** |
| `avg_peak_memory_gb` | 0.529 | 0.547 | +3.4% |
| `avg_perplexity` | 8.49 | 6.05 | **-28.7% (better)** |
| `quality_pass` | True | True | -- |

### Full Experiment Log

| # | Experiment | gen_tps | prompt_tps | ppl | Decision | Notes |
|---|---|---|---|---|---|---|
| 0 | **Baseline** (stream_generate, temp=0.7, top_p=0.9) | 397.0 | 2,467.7 | 8.49 | -- | Starting point |
| 1 | Argmax sampling (temp=0.0, top_p=1.0) | **429.2** | 2,996.4 | 6.05 | **KEEP** | +8.1% — removes all sampling overhead |
| 2 | Prefill step size 512 | 429.8 | 2,854.2 | 6.05 | REVERT | Within noise, prompt_tps dropped |
| 3 | Custom generate loop (generate_step directly) | **433.6** | 3,160.1 | 6.05 | **KEEP** | +1% — bypasses stream_generate overhead |
| 4 | Metal cache limit 1GB | **435.8** | 3,134.3 | 6.05 | **KEEP** | +0.5% — reduces buffer allocation churn |
| 5 | mx.async_eval in decode loop | 425.8 | 3,125.1 | 6.05 | REVERT | -2.3% — .item() forces sync anyway |
| 6 | Singleton sampler/EOS/memory caching | 429.2 | 3,299.0 | 6.05 | REVERT | -1.5% — indirection hurt more than caching helped |
| 7 | 8-bit KV cache quantisation | 361.7 | 2,594.8 | 4.48 | REVERT | -17% — quant/dequant overhead exceeds bandwidth savings on small model |
| 8 | MAX_TOKENS 128 | 435.5 | 3,170.2 | 6.44 | REVERT | gen_tps flat, memory better but not the target metric |
| 9 | Metal cache limit 4GB | 427.8 | 3,256.7 | 6.05 | REVERT | -1.8% — larger cache caused more GC pressure |
| 10 | Disable Python GC during generation | 430.9 | 3,279.9 | 6.05 | REVERT | -1.1% — GC not a bottleneck for small models |
| 11 | Pre-build KV cache before generation | **435.8** | 3,131.1 | 6.05 | **KEEP** | Tighter variance (432-438 vs 407-440) |
| 12 | Prefill step size 4096 | 435.2 | 3,180.8 | 6.05 | REVERT | No improvement over 2048 |
| 13 | Pre-materialize model weights on Metal | **436.1** | 3,135.5 | 6.05 | **KEEP** | Marginal but consistent |
| 14 | Fully custom forward loop (skip logprobs) | 339.7 | 3,438.3 | 6.05 | REVERT | -22% — lost generate_step's async pipelining |
| 15 | Streamlined decode + raw argmax sampler | **434.9** | 3,180.0 | 6.05 | **KEEP** | Same speed, -19 lines — simplicity wins |
| 16 | MAX_KV_SIZE=512 rotating cache | 434.5 | 3,347.4 | 6.05 | REVERT | Rotating cache overhead negated savings |

### Key Findings

1. **Biggest win: argmax sampling (+8.1%).** Switching from temp=0.7/top_p=0.9 to deterministic argmax was the single largest speedup. The sampling path (nucleus sampling, temperature scaling) has real overhead.

2. **generate_step's pipelining is excellent.** Our custom forward loop that skipped logprobs was 22% *slower* because it lost generate_step's `mx.async_eval` pipelining — it pre-computes token N+1 while yielding token N. Don't fight the framework.

3. **KV cache quantisation hurts small models.** 8-bit KV cache was 17% slower on 0.5B — the quantise/dequantise overhead vastly exceeds any memory bandwidth savings at this scale.

4. **Python-level optimisations have diminishing returns.** GC disabling, singleton caching, async eval changes — all within noise or harmful. The bottleneck is Metal GPU compute, not Python.

5. **Metal cache limit has a sweet spot.** 1GB helped slightly, 4GB hurt. Over-allocating the Metal buffer pool can increase GC pressure.

6. **Simpler code at same performance is a valid improvement.** The final inference.py is 30% shorter than intermediate versions while maintaining peak throughput.

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
