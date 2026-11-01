# Open auxiliary dataset v2 — workflow script index (archival)

This directory archives **existing** collection, screening, curation, acceptance, and export scripts from the original integration workspace. Scripts assume the original `work/` tree, HF caches, and server paths unless manually retargeted; they are not a universal fresh-clone reproduction interface.

## Pipeline scripts (`pipeline/`)

| Stage | Repo copy | Original workspace path |
| --- | --- | --- |
| Download verified HF parquet slice | `pipeline/download_verified_dataset_v2.py` | `work/download_verified_dataset_v2.py` |
| Parquet row reader (RangeFile / fetch helper; slow full-scan strategy retired) | `pipeline/read_published_parquet_candidates.py` | `work/read_published_parquet_candidates.py` |
| Additional CrewAI published rows | `pipeline/read_additional_crewai_published_rows.py` | `work/read_additional_crewai_published_rows.py` |
| Select + strict patch reconstruction | `pipeline/select_and_reconstruct_v2.py` | `work/select_and_reconstruct_v2.py` |
| CrewAI supplemental reconstruction | `pipeline/reconstruct_additional_crewai.py` | `work/reconstruct_additional_crewai.py` |
| Reviewed SWE-rebench collection driver | `pipeline/run_reviewed_rebench_collection.py` | `work/run_reviewed_rebench_collection.py` |
| Integrated export | `pipeline/export_integrated_open_dataset.py` | `work/export_integrated_open_dataset.py` |
| Build release tree + JSONL splits | `pipeline/build_open_dataset_v2.py` | `work/build_open_dataset_v2.py` |
| Root admit | `pipeline/root_admit_open_dataset_v2.py` | `work/root_admit_open_dataset_v2.py` |
| Root verify (main) | `pipeline/root_verify_open_dataset_v2.py` | `work/root_verify_open_dataset_v2.py` |
| Root verify (last gaps) | `pipeline/root_verify_last_gaps_v2.py` | `work/root_verify_last_gaps_v2.py` |
| Freeze missing parent interfaces | `pipeline/root_freeze_missing_parent_interfaces.py` | `work/root_freeze_missing_parent_interfaces.py` |
| Root final data decisions | `pipeline/write_root_final_data_decisions.py` | `work/write_root_final_data_decisions.py` |
| Seal manifest + card | `pipeline/seal_open_dataset_v2.py` | `work/seal_open_dataset_v2.py` |
| Diagnostic tokenizer-only check | `pipeline/check_open_dataset_tokenizer_only.py` | `work/check_open_dataset_tokenizer_only.py` |

Byte-identical subsets also live under the frozen dataset snapshot: `data/open_auxiliary/v2/data_build_scripts/` (seven build/verify/seal scripts recorded at release time).

## V1 rebench pilot

| Script | Repo copy | Original |
| --- | --- | --- |
| Pilot collector | `v1_rebench/collect_pilot.py` | `work/open_dataset_integration_v1/rebench/collect_pilot.py` |

## V1 manual curation assembly

| Script | Repo copy | Original |
| --- | --- | --- |
| Assemble curated examples | `v1_curated/assemble_curated_examples.py` | `work/open_dataset_integration_v1/curated/assemble_curated_examples.py` |

## V2 editorial Plan curation batches (manual / model-assisted)

**Plan writers / normalizers** (generate or assemble curated JSON — not autonomous teacher Plans):

| Batch | Generator(s) | Original |
| --- | --- | --- |
| DSPy | `v2_curation_batches/dspy_batch/write_curated_artifacts.py` | `work/open_dataset_integration_v2/dspy_batch/write_curated_artifacts.py` |
| CrewAI issues | `v2_curation_batches/crewai_batch/build_curated.py` | `work/open_dataset_integration_v2/crewai_batch/build_curated.py` |
| CrewAI PRs | `v2_curation_batches/crewai_pr_batch/build_curation.py` | `work/open_dataset_integration_v2/crewai_pr_batch/build_curation.py` |
| LiteLLM | `v2_curation_batches/litellm_batch/generate_curated_rows.py` | `work/open_dataset_integration_v2/litellm_batch/generate_curated_rows.py` |
| CrewAI strict normalizers | `v2_curation_batches/crewai_strict/build_curated.py`, `write_curation_decisions.py` | `work/open_dataset_integration_v2/crewai_strict/` |

**Verification drivers** (bounded checks / sync curated outputs — archival):

| Batch | Verification copy | Original |
| --- | --- | --- |
| DSPy batch | `v2_curation_batches/dspy_batch/verify_batch.py` | `dspy_batch/verify_batch.py` |
| CrewAI batch | `v2_curation_batches/crewai_batch/verify_batch.py` | `crewai_batch/verify_batch.py` |
| CrewAI PR batch | `v2_curation_batches/crewai_pr_batch/verify_batch.py` | `crewai_pr_batch/verify_batch.py` |
| LiteLLM batch | `v2_curation_batches/litellm_batch/verify_batch.py` | `litellm_batch/verify_batch.py` |
| CrewAI strict | `v2_curation_batches/crewai_strict/verify_batch.py` | `crewai_strict/verify_batch.py` |

Accepted/rejected drafts: `data/open_auxiliary/v2/curator_drafts/`. Strict portable verification: `data/open_auxiliary/v2/portable_proofs/*/verify_batch.py`.

## Frozen verification evidence (pointer only)

| Location in repo | Role |
| --- | --- |
| `data/open_auxiliary/v2/root_verification/` | Root before/after and preservation proof inputs |
| `data/open_auxiliary/v2/portable_proofs/` | Bounded behavior verification scripts + fixtures |
| `data/open_auxiliary/v2/portable_sources/` / `refs/` | Pinned source snapshots referenced by proofs |
| `data/open_auxiliary/v2/data_build_scripts/` | Scripts as frozen at seal time |
| `data/open_auxiliary/v2/third_party_licenses/`, `NOTICE.md` | Upstream LICENSE texts and compilation attribution |
