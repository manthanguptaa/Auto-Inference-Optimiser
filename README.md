# Auto-Inference-Optimiser

An autonomous agent that optimises LLM inference speed on Apple Silicon, inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

Point an AI coding agent at this repo and let it run experiments overnight — it hill-climbs on tokens/sec by modifying the inference pipeline, using git commits as experiment tracking.

## Results: Claude Opus 4.6 Optimisation Runs

### Run 3: 0.5B Model (with sanity checks)

**Hardware:** MacBook Pro M4, 48GB RAM
**Model:** `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (0.5B params, 4-bit)
**Agent:** Claude Opus 4.6 via Claude Code
**Experiments:** 11 total, 2 kept, 9 reverted

| Metric | Baseline | Optimised | Change |
|---|---|---|---|
| `avg_generation_tps` | 394.97 | 437.17 | **+10.7%** |
| `avg_prompt_tps` | 2,728.70 | 2,662.91 | -2.4% |
| `avg_peak_memory_gb` | 0.544 | 0.544 | 0% |
| `avg_perplexity` | 5.99 | 6.05 | +1.0% |
| `sanity_check` | 0.80 | 0.80 | 0% (4/5 tasks pass) |
| `quality_pass` | True | True | -- |

<details>
<summary>Full experiment log (0.5B)</summary>

| # | Experiment | gen_tps | prompt_tps | ppl | sanity | Decision | Tradeoff |
|---|---|---|---|---|---|---|---|
| 0 | **Baseline** (stream_generate, temp=0.7, top_p=0.9) | 394.97 | 2,728.70 | 5.99 | 0.80 | -- | Starting point — no tradeoffs |
| 1 | Argmax sampling (temp=0.0, top_p=1.0) | **437.55** | 2,600.19 | 6.05 | 0.80 | **KEEP** | +10.8% — deterministic output, no sampling diversity |
| 2 | Metal cache limit 1GB | 436.51 | 2,571.70 | 6.05 | 0.80 | REVERT | No improvement over exp1 (-0.2%) |
| 3 | Prefill step size 512 | 437.89 | 2,825.67 | 6.05 | 0.80 | REVERT | Within noise (+0.08%), smaller chunks add kernel launch overhead |
| 4 | 8-bit KV cache quantisation | 366.51 | 2,134.13 | 4.48 | **0.40** | REVERT | **Sanity gate FAILED** (2/5 tasks) — reduced precision broke output quality. Perplexity alone (4.48) would have passed |
| 5 | List join instead of string concat | 437.93 | 2,570.83 | 6.05 | 0.80 | REVERT | Within noise (+0.09%), no meaningful difference |
| 6 | 4-bit KV cache quantisation | 371.98 | 2,059.42 | 602.14 | **0.20** | REVERT | **Both gates FAILED** — catastrophic quality collapse, perplexity 602, only 1/5 tasks pass |
| 7 | Rotating KV cache 1024 tokens | 436.36 | 2,847.11 | 6.05 | 0.80 | REVERT | Slightly worse (-0.3%), model loses context beyond 1024 tokens |
| 8 | Disable Python GC during generation | 437.80 | 2,780.20 | 6.05 | 0.80 | REVERT | Within noise (+0.06%), adds complexity for no gain |
| 9 | Prefill step size 4096 | 437.05 | 2,805.26 | 6.05 | 0.80 | REVERT | Within noise (-0.1%), larger chunks use more peak memory |
| 10 | Minimal code — singleton sampler, inline format, remove unused config | **437.17** | 2,662.91 | 6.05 | 0.80 | **KEEP** | Same speed, 42 fewer lines — simplicity wins |
| 11 | MAX_TOKENS=128 | 440.01 | 2,625.50 | 6.44 | 0.80 | REVERT | Speed gain is artificial — model does less work, not faster work |

</details>

### Run 4: 3B Model — Llama 3.2 (with sanity checks)

**Hardware:** MacBook Pro M4, 48GB RAM
**Model:** `mlx-community/Llama-3.2-3B-Instruct-4bit` (3B params, 4-bit)
**Agent:** Claude Opus 4.6 via Claude Code
**Experiments:** 8 total, 2 kept, 6 reverted

| Metric | Baseline | Optimised | Change |
|---|---|---|---|
| `avg_generation_tps` | 115.55 | 118.10 | **+2.2%** |
| `avg_prompt_tps` | 728.98 | 775.44 | **+6.4%** |
| `avg_peak_memory_gb` | 2.314 | 2.314 | 0% |
| `avg_perplexity` | 4.12 | 4.38 | +6.3% |
| `sanity_check` | 0.87 | 0.80 | -8% (4/5 vs 13/15) |
| `quality_pass` | True | True | -- |

<details>
<summary>Full experiment log (Llama 3.2 3B)</summary>

| # | Experiment | gen_tps | prompt_tps | ppl | sanity | Decision | Tradeoff |
|---|---|---|---|---|---|---|---|
| 0 | **Baseline** (stream_generate, temp=0.7, top_p=0.9) | 115.55 | 728.98 | 4.12 | 0.87 | -- | Starting point — no tradeoffs |
| 1 | Argmax sampling (temp=0.0, top_p=1.0) | **118.58** | 754.40 | 4.38 | 0.80 | **KEEP** | +2.6% — deterministic output, no sampling diversity. Sanity dropped 0.87→0.80 (math now consistently fails with greedy decoding) |
| 2 | Metal cache limit 2GB | 118.32 | 737.89 | 4.38 | 0.80 | REVERT | Within noise (-0.2%), memory tuning doesn't help at this scale |
| 3 | 8-bit KV cache quantisation | 112.00 | 695.77 | 4.38 | 0.80 | REVERT | -5.5% speed — quant/dequant overhead exceeds bandwidth savings |
| 4 | 4-bit KV cache quantisation | 111.39 | 697.67 | 3.91 | 0.80 | REVERT | -6.1% speed — Llama handles 4-bit KV better than Qwen (no quality collapse) but still slower |
| 5 | Rotating KV cache 1024 tokens | 118.36 | 749.25 | 4.38 | 0.80 | REVERT | Within noise (-0.2%), model forgets beyond window |
| 6 | Prefill step size 512 | 118.04 | 761.34 | 4.38 | 0.80 | REVERT | Within noise (-0.5%), smaller chunks add kernel launch overhead |
| 7 | Minimal code — singleton sampler, inline format, remove unused config | **118.10** | 775.44 | 4.38 | 0.80 | **KEEP** | Same speed, 42 fewer lines — simplicity wins |

</details>

### Key Findings

1. **Argmax sampling is universally the biggest win.** +10.8% on Qwen 0.5B, +2.6% on Llama 3B. Nucleus sampling (top-p) has real per-token overhead. Tradeoff: deterministic output, no diversity.

2. **KV cache quantisation hurts — and the sanity check catches what perplexity misses.** On Qwen 0.5B, 8-bit KV dropped sanity_check to 0.40 (2/5 tasks pass) while perplexity stayed at 4.48 (would have passed the 50.0 threshold). 4-bit was catastrophic: perplexity 602, sanity_check 0.20. On Llama 3B, 4-bit KV didn't collapse quality (sanity 0.80) but still lost 6.1% speed. Architecture matters for quantization tolerance.

3. **The memory bandwidth wall is real.** Both 3B models (~118 tok/s) and the 0.5B model (~437 tok/s) are near the M4's bandwidth ceiling. Most experiments landed within measurement noise.

4. **Results are consistent across model families.** Qwen and Llama show the same patterns: argmax wins, KV quantization loses speed, config tuning is noise, simplification is free.

5. **Simplicity wins.** Removing 42 lines of unused config maintained the same speed on both models. Less code = less to review.

6. **MAX_TOKENS reduction is artificial.** The change monitor correctly flagged this — speed appeared to improve only because the model generated fewer tokens, not faster.

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
│  6. Check: quality_pass (perplexity + sanity_check)?        │
│  7. If both YES → keep. If NO → git reset --hard HEAD~1    │
│  8. Log results to results.tsv                              │
│  9. Go to 1.                                                │
└─────────────────────────────────────────────────────────────┘
```

When `prepare.py` runs, it also prints a **change monitor report** showing what the agent modified and any tradeoff warnings:

```
============================================================
CHANGES FROM BASELINE
============================================================
  TEMP: 0.7 -> 0.0

Tradeoff warnings:
  - Argmax decoding: faster (no sampling) but deterministic, no diversity.
============================================================
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
| `avg_perplexity` | Must stay below 50.0 | Quality gate — model-internal coherence score |
| `sanity_check` | Must stay above 0.6 | Quality gate — task-level correctness (see below) |

### Quality Gates

An experiment is automatically reverted if **either** gate fails:

- **Perplexity gate** (`avg_perplexity < 50.0`): Catches catastrophic quality collapse (e.g., 4-bit KV quantization spiked perplexity to 69,405).
- **Sanity check gate** (`sanity_check >= 0.6`): Checks that outputs are actually correct — the math answer is right, the explanation mentions key concepts, the code contains a function definition. Each benchmark prompt has a content check; the score is the fraction that pass.

The sanity checks for each benchmark prompt:
| Prompt | Check |
|---|---|
| Transformers explanation | Mentions >= 2 of: attention, self-attention, token, layer |
| Compiler optimization summary | Mentions >= 2 optimization passes (DCE, inlining, etc.) |
| Train speed problem | Answer contains "48" (the correct answer: 48 km/h) |
| Silicon/electricity poem | Multi-line + relevant keywords (silicon, chip, electric, etc.) |
| LCS code generation | Contains `def` + relevant identifiers (subsequence, lcs, etc.) |

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

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~4GB free RAM (for 3B model) or ~2GB (for 0.5B model)
