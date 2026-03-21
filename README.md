# Auto-Inference-Optimiser

An autonomous agent that optimises LLM inference speed on Apple Silicon, inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

Point an AI coding agent at this repo and let it run experiments overnight — it hill-climbs on tokens/sec by modifying the inference pipeline, using git commits as experiment tracking.

## Results: Claude Opus 4.6 Optimisation Runs

### Run 1: 0.5B Model (main branch)

**Hardware:** MacBook Pro M4, 48GB RAM
**Model:** `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (0.5B params, 4-bit)
**Agent:** Claude Opus 4.6 via Claude Code
**Experiments:** 16 total, 5 kept, 11 reverted

| Metric | Baseline | Optimised | Change |
|---|---|---|---|
| `avg_generation_tps` | 397.0 | 434.9 | **+9.5%** |
| `avg_prompt_tps` | 2,467.7 | 3,180.0 | **+28.9%** |
| `avg_peak_memory_gb` | 0.529 | 0.547 | +3.4% |
| `avg_perplexity` | 8.49 | 6.05 | **-28.7% (better)** |
| `quality_pass` | True | True | -- |

<details>
<summary>Full experiment log (0.5B)</summary>

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

</details>

### Run 2: 3B Model (this branch)

**Hardware:** MacBook Pro M4, 48GB RAM
**Model:** `mlx-community/Qwen2.5-3B-Instruct-4bit` (3B params, 4-bit)
**Agent:** Claude Opus 4.6 via Claude Code
**Experiments:** 16 total, 3 kept, 13 reverted

| Metric | Baseline | Optimised | Change |
|---|---|---|---|
| `avg_generation_tps` | 115.96 | 119.33 | **+2.9%** |
| `avg_prompt_tps` | 634.68 | 628.01 | -1.1% |
| `avg_peak_memory_gb` | 2.278 | 2.278 | 0% |
| `avg_perplexity` | 6.47 | 4.55 | **-29.7% (better)** |
| `quality_pass` | True | True | -- |

<details>
<summary>Full experiment log (3B)</summary>

| # | Experiment | gen_tps | prompt_tps | mem_gb | ppl | Decision | Notes |
|---|---|---|---|---|---|---|---|
| 0 | **Baseline** (stream_generate, temp=0.7, top_p=0.9) | 115.96 | 634.68 | 2.278 | 6.47 | -- | Starting point |
| 1 | Argmax sampling (temp=0.0, top_p=1.0) | **118.86** | 625.51 | 2.278 | 4.55 | **KEEP** | +2.5% — removes sampling overhead |
| 2 | Custom generate loop (generate_step) | 105.6 | 77.8 | 1.768 | 4.55 | REVERT | -8.9% — manual timing conflates prefill+decode |
| 3 | Metal cache limit 1GB | 118.94 | 625.08 | 2.278 | 4.55 | REVERT | Within noise (+0.07%) |
| 4 | Metal cache limit 2GB | **119.03** | 637.34 | 2.278 | 4.55 | **KEEP** | Marginal gen_tps, +1.9% prompt_tps |
| 5 | Prefill step size 512 | 119.24 | 627.28 | 2.278 | 4.55 | REVERT | Within noise, prompt_tps dropped |
| 6 | Prefill step size 4096 | 119.14 | 628.77 | 2.278 | 4.55 | REVERT | Within noise |
| 7 | 8-bit KV cache quantisation | 109.88 | 586.35 | 2.278 | 4.32 | REVERT | -7.7% — quant/dequant overhead exceeds bandwidth savings |
| 8 | MAX_KV_SIZE=1024 rotating cache | 119.25 | 632.41 | 2.278 | 4.55 | REVERT | Within noise (+0.18%) |
| 9 | Minimal code (singleton sampler, join) | **119.33** | 628.01 | 2.278 | 4.55 | **KEEP** | Same speed, 60% less code |
| 10 | Metal cache limit 4GB | 119.45 | 610.24 | 2.278 | 4.55 | REVERT | prompt_tps dropped |
| 11 | Remove Metal cache limit | 119.33 | 614.04 | 2.278 | 4.55 | REVERT | prompt_tps dropped vs 2GB limit |
| 12 | Disable Python GC during generation | 119.21 | 622.58 | 2.278 | 4.55 | REVERT | GC not a bottleneck |
| 13 | 4-bit KV cache quantisation | 120.51 | 595.26 | 2.266 | 69,405 | REVERT | **Quality gate FAILED** — perplexity exploded |
| 14 | Pre-materialize model weights | 118.32 | 638.74 | 2.278 | 4.55 | REVERT | -0.8% — extra mx.eval overhead |
| 15 | Prefill step size 1024 | 118.77 | 637.90 | 2.278 | 4.55 | REVERT | Within noise |
| 16 | MAX_TOKENS=128 | 119.08 | 634.50 | 2.116 | 4.79 | REVERT | gen_tps within noise |

</details>

### Key Findings: 0.5B vs 3B

1. **Smaller models have more optimisation headroom.** The 0.5B model gained +9.5% while the 3B model only gained +2.9%. The 3B model at ~119 tok/s is already at ~87% of the M4's theoretical memory bandwidth ceiling (~273 GB/s), leaving almost no room for software optimisation.

2. **Argmax sampling is universally the biggest win.** +8.1% on 0.5B, +2.5% on 3B. Nucleus sampling (top-p) has real per-token overhead that scales with vocab size.

3. **KV cache quantisation hurts at all scales tested.** 8-bit was -17% on 0.5B and -7.7% on 3B. 4-bit on 3B destroyed output quality (perplexity 69,405). The quant/dequant kernels add more overhead than they save in bandwidth for 4-bit weight models.

4. **Custom generate loops are dangerous.** On 0.5B, bypassing stream_generate for generate_step directly gave +1%. On 3B, the same approach was -8.9% because manual timing conflated prefill and decode phases. The framework's internal metrics are more accurate.

5. **The memory bandwidth wall is real.** A 3B 4-bit model is ~2GB. At 119 tok/s, that's 238 GB/s — close to the M4's theoretical 273 GB/s. No amount of Python-level optimisation can push past this hardware limit.

6. **Metal cache limit has a sweet spot per model.** 1GB was optimal for 0.5B (~350MB model), 2GB was optimal for 3B (~2GB model). The cache should roughly match the model size.

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

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~4GB free RAM (for 3B model) or ~2GB (for 0.5B model)
