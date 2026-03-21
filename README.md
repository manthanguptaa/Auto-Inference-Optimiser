# Auto-Inference-Optimiser

An autonomous agent that optimises LLM inference speed on Apple Silicon, inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

Point an AI coding agent at this repo and let it run experiments overnight — it hill-climbs on tokens/sec by modifying the inference pipeline, using git commits as experiment tracking.

## Results: Claude Opus 4.6 Optimisation Runs

**Hardware:** MacBook Pro M4, 48GB RAM
**Agent:** Claude Opus 4.6 via Claude Code

We ran two rounds of experiments — first on a 0.5B model (config-tuning), then restructured the project and ran on a 3B model (deep inference internals).

---

### Round 1: Qwen2.5-0.5B-Instruct-4bit (config tuning)

**16 experiments, 5 kept, 11 reverted**

| Metric | Baseline | Optimised | Change |
|---|---|---|---|
| `avg_generation_tps` | 397.0 | 434.9 | **+9.5%** |
| `avg_prompt_tps` | 2,467.7 | 3,180.0 | **+28.9%** |
| `avg_peak_memory_gb` | 0.529 | 0.547 | +3.4% |
| `avg_perplexity` | 8.49 | 6.05 | **-28.7% (better)** |

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

---

### Round 2: Qwen2.5-3B-Instruct-4bit (deep inference internals)

For Round 2, we restructured the project to inline the full inference pipeline (KV cache, generate loop, attention) into `inference.py`, giving the agent a much deeper attack surface. We also switched to a 3B model that's actually memory-bandwidth bound.

**5 experiments, 2 kept, 3 reverted**

| Metric | Baseline | Optimised | Change |
|---|---|---|---|
| `avg_generation_tps` | 118.4 | 118.5 | +0.1% |
| `avg_prompt_tps` | 612.1 | 630.4 | **+3.0%** |
| `avg_peak_memory_gb` | 2.262 | 2.262 | 0% |
| `avg_perplexity` | 4.55 | 4.55 | 0% |

<details>
<summary>Full experiment log (3B)</summary>

| # | Experiment | gen_tps | prompt_tps | ppl | Decision | Notes |
|---|---|---|---|---|---|---|
| 0 | **Baseline** (inlined pipeline, argmax) | 118.4 | 612.1 | 4.55 | -- | Starting point |
| 1 | KV cache step size 512 | 118.6 | 602.1 | 4.55 | REVERT | Within noise |
| 2 | Metal cache 4GB + prefill step 1024 | 118.5 | 588.1 | 4.55 | REVERT | prompt_tps dropped |
| 3 | Skip logprobs (argmax on raw logits) | **118.7** | 589.8 | 4.55 | **KEEP** | Cleaner code, removes logsumexp over 151k vocab |
| 4 | Remove generation stream + clear_cache | **118.5** | **630.4** | 4.55 | **KEEP** | +3% prefill from removing sync overhead |
| 5 | Speculative decoding (0.5B draft) | 58.0 | 605.1 | 7.27 | REVERT | -51% — draft/target disagreement + double memory reads |

</details>

---

### Key Findings

**What worked:**
1. **Argmax sampling was the biggest single win (+8.1% on 0.5B).** Nucleus sampling has real overhead — the temp scaling, probability sorting, and cumsum are expensive.
2. **Bypassing framework overhead helps on small models.** Using `generate_step` directly instead of `stream_generate` saved ~1% by avoiding per-token response object creation.
3. **Removing unnecessary computation matters.** Skipping logprobs (logsumexp over 151k vocab) and removing stream context managers reduced overhead.

**What didn't work (and why):**
1. **Speculative decoding was 51% slower.** On unified memory, running two models (0.5B + 3B) sequentially doubles memory bandwidth usage. The draft/target disagreement rate was too high for the math to work.
2. **KV cache quantisation was 17% slower on 0.5B.** The quantise/dequantise overhead vastly exceeds bandwidth savings on small models. Would likely help on >7B models.
3. **Python-level optimisations (GC disable, singletons) had no effect.** The bottleneck is Metal GPU compute, not Python.
4. **The 3B model hit a hard memory bandwidth wall at ~118 tok/s.** At 4-bit quantisation, the model is ~2GB. Reading all weights per token at 118 tok/s = ~236 GB/s, close to the M4's theoretical bandwidth. No software trick can bypass physics.

**The meta-lesson:** The autoresearch pattern works best when the search space is deep (modifiable architecture, not just config). But even with deep access, hardware limits set a hard ceiling. The most interesting finding isn't any single optimisation — it's learning exactly where the wall is and why.

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
| `inference.py` | Full inference pipeline (KV cache, generate loop, attention) | **Yes** |
| `program.md` | Agent instructions | No |

## The Optimisation Search Space

In v2, `inference.py` contains inlined implementations the agent can modify:

- **KVCache class** — allocation strategy, step size, quantisation, sliding windows
- **scaled_dot_product_attention()** — attention computation, sparse patterns
- **_generate_step()** — prefill chunking, decode pipelining, async eval strategy
- **Configuration** — Metal cache limits, prefill step size, max tokens
- **Speculative decoding** — draft model integration, verification logic

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

```bash
# Example with Claude Code
claude --system-prompt program.md
```

## Benchmark Models

- **v1:** `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (0.5B, ~350MB) — compute-bound
- **v2:** `mlx-community/Qwen2.5-3B-Instruct-4bit` (3B, ~2GB) — memory-bandwidth-bound

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~4GB free RAM (for 3B model)
