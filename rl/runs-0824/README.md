# Bench records — 2026-08-24

Zero-shot heads-up NLHE, played through the `act` tool interface (rl/toolcall.py),
temperature 1.0, thinking disabled. bb/100 from seat A's perspective; fish evals use
mirrored decks. Raw logs in this directory.

## Canonical zero-shot ladder — Qwen3.8-27B (bf16, vLLM on 1×H200)

768-token reply budget, single-shot format (one system+observation exchange per decision).

| opponent | bb/100 | hands | invalid | note |
|---|---|---|---|---|
| nit | +62 | 200 | 0.0% | near the ≈+75 exploit ceiling |
| station | +338 ± 90 | 800 | ≤0.8% | |
| maniac | **−72 ± 155** | 1,600 | ≤0.9% | breakeven — no sign-flip training story |
| Slumbot | **−206** (VR; −64 raw) | 250 | 0.4% | second session; first session VR −169 |
| random | inconclusive | 200 | — | needs thousands of hands |

Slumbot reference points: unscaffolded frontier ≈ −160, scaffolded ≈ −57, solver 0.
Zero-shot 27B ≈ the unscaffolded-frontier line.

## Incident log (what the numbers cost to get right)

1. **Token-budget eval artifact.** The first ladder ran with `max_new_tokens=128`.
   Rare truncations (2–6% invalid) concentrated on long-reply big-pot decisions;
   each became a forced default FOLD. Controlled A/B, same seed, 200 hands vs maniac:
   128 tokens → −431; 768 tokens → +43. ~470 bb/100 of bias from a decoding flag.
   Every v1 number (`base-*.log`) is superseded by v2 (`v2-*.log`).
2. **Format sensitivity (open).** Training-thread format (multi-turn tool thread,
   system text in the first user message) read ≈ +1,100 bb/100 vs maniac at
   near-zero policy movement — far above the single-shot eval. Untangling format
   effect from 64-episode noise needs a thread-format eval mode (`play.py`, TODO).
3. **Opus 5 run lost ($10.92, zero rows).** OpenRouter eval launched on an
   estimated cost/hand that was ~3× low (verbose analysis text billed at $25/M out),
   with results held in buffered stdout until process exit. Spend hit $10.55 before
   the first check; killing the workers destroyed the unflushed data.
   Fixes now in tree: flushed per-hand progress + live token meter, `--token-budget`
   clean stop with partial stats, `--terse`. Protocol: 5-hand paid probe → read
   measured $/hand from the provider's usage endpoint → size the run → one opponent
   at a time.

Day's spend: RunPod $12.76 (all infrastructure lessons + both ladders + ~40 min of
27B training), OpenRouter $10.92 (incident 3).
