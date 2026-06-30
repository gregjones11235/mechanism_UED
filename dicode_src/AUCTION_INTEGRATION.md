# Auction (multi-FM) integration — what changed vs baseline DiCode

This is a vendored copy of DiCode (`src/dicode`, `src/minicraftax`) plus the offline `auction/`
module (the multi-FM curriculum method). The auction layer is wired into the ②description layer
ONLY; everything else (parent selection, code gen, compilation, PPO training, reward, archive) is
untouched. See `../v1_experiment.md` §7 and `../方法设计_v1.md` for the design.

## Files added / changed

| File | Change |
|---|---|
| `auction/` | The offline method module (Proposal, Coverage+submodularity, Endorsement, AmbitionGain, GreedyTopK selector; v2: PriceState, WalrasianSelector, contest). 65 unit tests. |
| `src/dicode/dreaming/auction_integration.py` | **NEW.** Adapter: parses `Relevant Achievements:` from a docstring → lowercase 67-achievement set; converts DiCode `parsed_response` ↔ `auction.Proposal`. |
| `src/dicode/dreaming/gen_manager.py` | **Additive only.** `TaskGenerator.__init__` takes optional `proposer_llms`; new methods `evolve_mastered_auction` + `_build_mastered_prompts` (extracted from `evolve_mastered`, which is byte-unchanged); `GenManager.__init__` builds N proposer LLMs from config; `evolve_tasks` switches on `config.gen_manager.auction`. |
| `pyproject.toml` | `auction` added to wheel packages so `pip install -e .` exposes `import auction`. |

## Baseline safety (clean ablation)

With NO auction config keys present, behaviour is **byte-identical to baseline DiCode**:
- `config.gen_manager.auction` absent → `.get("auction", False)` = False → calls original `evolve_mastered`.
- `config.gen_manager.proposers` absent → `proposer_llms = None` → `TaskGenerator` uses `[task_designer]` (single FM).

So `auction=false` (or unset) == DiCode (= the B-arm baseline). N=1 + k=full ≈ baseline too.

## How to turn the method ON (config)

Add to the `gen_manager` config group:

```yaml
gen_manager:
  auction: true            # route to evolve_mastered_auction
  auction_k: 10            # how many descriptions to keep after the auction (default = #parents)
  proposers:               # N heterogeneous Proposers (omit to reuse task_generator as the sole proposer)
    - { provider: deepinfra, base_url: ..., model: Qwen/Qwen3-235B-A22B-Thinking-2507, llm_type: ..., max_tokens: 32768, temperature: 0.6, top_p: 0.95, think: true }
    - { provider: deepinfra, base_url: ..., model: deepseek-ai/DeepSeek-V3, ... }
    - { provider: deepinfra, base_url: ..., model: <third base>, ... }
```

- **A-arm (same-model auction):** `proposers` = N copies of the SAME model with different persona system prompts.
- **C-arm (heterogeneous auction):** `proposers` = N different base models (above).

## Run tests (offline, no GPU)

```bash
cd dicode_src && python -m pytest auction/tests/ -q   # 65 passed
```

## Status (2026-06-30)

- ✅ auction module (v1+v2) complete, 65 tests green.
- ✅ wired into evolve_mastered_auction; baseline path proven unchanged; dataflow smoke-tested offline.
- ⬜ NOT yet run on Oscar with real LLMs (next: add the proposers config + a short go/no-go job on the 2nd GPU quota, parallel to the reproduction baseline job 3575920).
- ⬜ v2 (WalrasianSelector) not yet wired into evolve_mastered_auction — currently v1 GreedyTopK only. v2 wiring is a one-line selector swap when ready.
