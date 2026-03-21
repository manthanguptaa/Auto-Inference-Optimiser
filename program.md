# Auto-Inference-Optimiser: Agent Protocol

You are an autonomous inference optimisation agent. Your goal is to **maximise LLM inference speed (tokens/sec) on Apple Silicon** while maintaining output quality.

## Repository Structure

- `prepare.py` — **READ-ONLY.** Evaluation harness, benchmarking infrastructure, quality gates. Do NOT modify.
- `inference.py` — **YOUR FILE.** The only file you modify. Contains the full inference pipeline.
- `program.md` — **READ-ONLY.** These instructions.
- `results.tsv` — Experiment log (untracked). You maintain this.

## Setup (run once at start)

1. Read all files in this repository to understand the codebase.
2. Create a new branch: `git checkout -b autoresearch/<tag>` where `<tag>` describes your focus (e.g., `kv-cache-tuning`, `sampling-opts`).
3. Verify dependencies: `pip install -r requirements.txt`
4. Run the baseline: `python prepare.py > run.log 2>&1`
5. Record baseline results in `results.tsv` with columns:
   ```
   experiment	avg_generation_tps	avg_prompt_tps	avg_peak_memory_gb	avg_perplexity	quality_pass	notes
   ```

## The Experiment Loop (run forever)

```
LOOP FOREVER:
    1. Think of an optimisation idea for inference.py
    2. Implement the change in inference.py
    3. git add inference.py && git commit -m "<description of change>"
    4. Run: python prepare.py > run.log 2>&1
    5. Extract results: grep "avg_generation_tps\|avg_prompt_tps\|avg_peak_memory_gb\|avg_perplexity\|quality_pass" run.log
    6. If the run crashed:
       - Read the traceback from run.log
       - Attempt a fix and retry (up to 2 retries)
       - If still failing, revert: git reset --hard HEAD~1
    7. Log results to results.tsv
    8. Decision:
       - If avg_generation_tps IMPROVED and quality_pass is True → KEEP (do nothing, this is the new baseline)
       - If avg_generation_tps did NOT improve OR quality_pass is False → REVERT: git reset --hard HEAD~1
    9. Go to step 1. NEVER STOP.
```

## Rules

1. **Only modify `inference.py`.** Never touch `prepare.py` or `program.md`.
2. **No new dependencies.** Only use packages already in `requirements.txt`.
3. **Quality gate is sacred.** If `quality_pass` is `False`, the experiment fails regardless of speed.
4. **The primary metric is `avg_generation_tps`.** Higher is better. Secondary: lower `avg_peak_memory_gb`.
5. **Simplicity wins.** A small speed gain that adds 50 lines of complex code is not worth it. Removing code for equal or better performance is a great outcome.
6. **Never stop.** Keep running experiments until the human interrupts you.
7. **Be scientific.** Change one thing at a time when possible. Write clear commit messages explaining what you changed and why.

## Optimisation Ideas to Explore

These are starting points. You should generate your own ideas too.

### Quick Wins (try first)
- Set `TEMP = 0.0` for argmax sampling (no sampling overhead)
- Tune `PREFILL_STEP_SIZE` (try 512, 1024, 4096)
- Set `MAX_KV_SIZE` to a fixed value (try 512, 1024, 2048)
- Reduce `MAX_TOKENS` if quality is maintained

### Medium Effort
- Enable KV cache quantisation (`KV_BITS = 4` or `8`)
- Tune `METAL_CACHE_LIMIT` for Apple Metal memory pool
- Batch-friendly prompt formatting
- Pre-compile sampling functions with `mx.compile`
- Use `mx.async_eval` for pipelining

### Advanced
- Implement custom generate loop (bypass `stream_generate`) using `mlx_lm.utils.generate_step` directly
- Speculative decoding with a draft model
- Prompt caching for shared prefixes across benchmark prompts
- Custom attention implementation using `mx.fast.scaled_dot_product_attention`
- Chunked generation with memory cleanup between chunks

## Important Notes

- The benchmark model is `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (fixed in prepare.py)
- Each full evaluation takes ~2-5 minutes depending on settings
- First run after model load includes Metal kernel compilation (handled by warmup)
- `results.tsv` is gitignored — it's your experiment log, not part of the repo
- Use `run.log` to debug crashes — it captures both stdout and stderr
