# Local adaptations vs upstream ModularRSI (A13)

## What we call upstream as-is
- `bash scripts/evolve.sh` with `PROFILE=smoke` → `python -m harbor.agents.terminus_2_modular.self_evo.phase0`
- Algorithmic Phase0 gates unchanged; smoke only scales workload (max_tasks=2, epochs=1, …)
- Solver temperature for P2A follows upstream Phase0 constant `0.0`

## Local adaptations (do not claim as upstream)
- Provider selection via this repo's `scripts/with_llm_provider.sh` / `llm_provider.py`
- Mandatory host+container proxy `http://172.17.0.1:7890` (Harbor in this ModularRSI pin lacks `--extra-docker-compose`)
- `SUPPORT_DATASET_DIR` → `/home/jinxu/arms_cache/modularrsi/ModularRSI_2000_Instances/tb`
- Archive root redirected under this research repo `runs/modularrsi_*` (not upstream `self_evo_runs/`)
- `LOCKED_MODULE` comes from research1 calibrate lock, not hardcoded `tools`
- H0 hash verification against `configs/h0.snapshot.v0.json` / `configs/research1/h0.lock.json`

## Upstream gaps still present
- No built-in research1 parent-step / long-horizon return labels
- No provider weekly ledger
- Cross-batch held-out protocol is research1's own requirement

## Non-claims
- Branch (P2B) samples are NOT counted as official ModularRSI reproduction evidence
