# Auto-Inference-Optimiser: Agent Protocol

You are an autonomous inference optimisation agent. Your goal is to **maximise LLM inference speed (tokens/sec) on Apple Silicon** while maintaining output quality.

## Repository Structure

- `prepare.py` — **READ-ONLY.** Evaluation harness, benchmarking, quality gates. Do NOT modify.
- `inference.py` — **YOUR FILE.** Contains the full inference pipeline with inlined internals. Modify freely.
- `program.md` — **READ-ONLY.** These instructions.
- `results.tsv` — Experiment log (untracked). You maintain this.

## What You Can Modify

`inference.py` contains inlined implementations of:

1. **KVCache class** — the data structure storing keys/values for each layer. You can change allocation strategy, step size, add quantisation, implement sliding windows, or replace entirely.

2. **scaled_dot_product_attention()** — currently calls `mx.fast.scaled_dot_product_attention`. You can experiment with custom attention patterns, sparse attention, or different implementations.

3. **_generate_step()** — the full generation loop with prefill and decode phases. You can change prefill chunking, async pipelining strategy, memory cleanup frequency, and sampling.

4. **Configuration** — MAX_TOKENS, PREFILL_STEP_SIZE, METAL_CACHE_LIMIT, and the sampler function.

5. **generate_text()** — the entry point. You can restructure tokenization, caching, or add prompt caching.

## The Metric

**Primary:** `avg_generation_tps` (higher is better)
**Quality gate:** `avg_perplexity < 50.0` (must pass)
**Secondary:** `avg_peak_memory_gb` (lower is better)

## Setup (run once)

1. Read all files in this repository.
2. Create a branch: `git checkout -b autoresearch/<tag>`
3. Run baseline: `python prepare.py > run.log 2>&1`
4. Record baseline in `results.tsv`:
   ```
   experiment	avg_generation_tps	avg_prompt_tps	avg_peak_memory_gb	avg_perplexity	quality_pass	notes
   ```

## The Experiment Loop (run forever)

```
LOOP FOREVER:
    1. Think of an optimisation idea
    2. Implement it in inference.py
    3. git add inference.py && git commit -m "<description>"
    4. python prepare.py > run.log 2>&1
    5. Extract results from run.log
    6. If crashed → read traceback, fix or revert (up to 2 retries)
    7. Log to results.tsv
    8. If avg_generation_tps improved AND quality_pass → KEEP
       Otherwise → git reset --hard HEAD~1
    9. Go to 1. NEVER STOP.
```

## Rules

1. **Only modify `inference.py`.** Never touch `prepare.py` or `program.md`.
2. **No new dependencies.** Only use packages in `requirements.txt`.
3. **Quality gate is sacred.** If `quality_pass` is `False`, the experiment fails.
4. **Be scientific.** Change one thing at a time. Write clear commit messages.
5. **Simplicity wins.** Removing code for equal results is a great outcome.
6. **Never stop.** Keep running experiments until interrupted.

## Model Context

- **Model:** `Qwen2.5-3B-Instruct-4bit` (~2GB, 36 transformer layers)
- **Hardware:** MacBook with Apple Silicon (Metal GPU, unified memory)
- **Bottleneck:** Memory bandwidth during decode (reading KV cache + model weights per token)
- The model is loaded by `prepare.py` using `mlx_lm.load()` — weights match the original Qwen2 architecture
- `inference.py` provides its own KVCache and generate loop — the model calls these via `cache=` parameter
