# Open auxiliary dataset collection (research1-open-auxiliary-integrated-v2)

Date organized into repo: 2026-11-02. **Auxiliary editorial Plan demonstrations only** — not formal autonomous SFT, not MC/G/D/U labels. Training and inference remain paused (`formal_autonomous_sft`: 0, `mc_values`: 0).

## Authoritative payload

| Item | Repository path |
| --- | --- |
| Dataset root | [`data/open_auxiliary/v2/`](../data/open_auxiliary/v2/) |
| Human-readable card | [`data/open_auxiliary/v2/README.md`](../data/open_auxiliary/v2/README.md) |
| Sealed content manifest (289 payload paths) | [`data/open_auxiliary/v2/dataset_manifest.json`](../data/open_auxiliary/v2/dataset_manifest.json) |
| Compilation / upstream NOTICE | [`data/open_auxiliary/v2/NOTICE.md`](../data/open_auxiliary/v2/NOTICE.md) |
| Immutable sealed manifest archive | [`data/open_auxiliary/v2/archive/sealed_dataset_manifest_release_v2.json`](../data/open_auxiliary/v2/archive/sealed_dataset_manifest_release_v2.json) |
| Repo packaging notes (no byte changes) | [`data/open_auxiliary/v2/repo_packaging_manifest.json`](../data/open_auxiliary/v2/repo_packaging_manifest.json) |
| Workflow script index | [`scripts/open_data/INDEX.md`](../scripts/open_data/INDEX.md) |
| Status snapshot (this import) | [`records/2026-11-02/open_data_status.json`](../records/2026-11-02/open_data_status.json) |

**Counts (unchanged from root approval):** 23 auxiliary **edit** samples (20 train / 3 validation); keep 0; 4 source repos; 23 independent parent code states; raw source ledger 96 rows (14 v1 rebench + 82 v2 candidates); 66 strict patch reconstructions; ledger disposition 63 not admitted, 7 pending verification, 2 verification failed, 1 reconstruction failed, 23 accepted auxiliary.

**JSONL layout:** Ten export JSONL files at dataset root (merged + per-partition train/validation/edit/keep sidecars). Four additional JSONL files under `source_record_corpus/` hold the raw 96-record corpus slices. Duplicate merged vs partition views are intentional; see `repo_packaging_manifest.json`.

## Stage map (actual scripts — not redesigned)

### 1. Source acquisition and screening

- **Verified HF parquet download:** `scripts/open_data/pipeline/download_verified_dataset_v2.py` (original `work/download_verified_dataset_v2.py`).
- **Candidate row reads:** `read_published_parquet_candidates.py` (RangeFile helper retained; full parquet scan strategy retired — archival only), `read_additional_crewai_published_rows.py`.
- **V1 SWE-rebench pilot collection:** `scripts/open_data/v1_rebench/collect_pilot.py` driven by `run_reviewed_rebench_collection.py`.
- **Outputs in snapshot:** `source_record_corpus/*.jsonl`, per-record `source_records/*.json`, `source_cards.json`.

### 2. Pinned before state and strict patch reconstruction

- **Selection + reconstruction:** `select_and_reconstruct_v2.py`, `reconstruct_additional_crewai.py`.
- **Frozen reconstructions:** `portable_sources/`, `refs/`, linked from `provenance.json` and `acceptance_ledger.json`.
- **Failures retained:** non-admitted and failed rows remain in corpus JSONL and ledger (not dropped).

### 3. Editorial Plan curation (manual / model-assisted, not autonomous teacher)

- **V1 curated assembly:** `scripts/open_data/v1_curated/assemble_curated_examples.py`.
- **V2 Plan writers:** `scripts/open_data/v2_curation_batches/dspy_batch/write_curated_artifacts.py`, `crewai_batch/build_curated.py`, `crewai_pr_batch/build_curation.py`, `litellm_batch/generate_curated_rows.py`, `crewai_strict/build_curated.py`, `crewai_strict/write_curation_decisions.py`.
- **V2 verification drivers (not Plan writers):** `scripts/open_data/v2_curation_batches/*/verify_batch.py` — see [`scripts/open_data/INDEX.md`](../scripts/open_data/INDEX.md).
- **Drafts accepted into export:** `curator_drafts/accepted/*.json`, review metadata `curation_review.json`.
- **Attribution:** `NOTICE.md`, `third_party_licenses/`, `resolved_license_inventory.json` (see also `source_cards.json`).

### 4. Bounded behavior and preservation proofs

- **Strict batch proofs:** `portable_proofs/*/verify_batch.py` + fixtures.
- **Root verification tree:** `root_verification/` (before/after probes, preservation evidence).
- **Scope extras:** `root_verification/curator_scope_extras/verify_extras.py` (frozen copy in snapshot).

### 5. Root admit, verify, freeze, export

- **Build integrated tree:** `build_open_dataset_v2.py` (also `data/open_auxiliary/v2/data_build_scripts/build_open_dataset_v2.py` at seal).
- **Root gates:** `root_admit_open_dataset_v2.py`, `root_verify_open_dataset_v2.py`, `root_verify_last_gaps_v2.py`, `root_freeze_missing_parent_interfaces.py`, `write_root_final_data_decisions.py`.
- **Export helper:** `export_integrated_open_dataset.py`.
- **Seal + manifest:** `seal_open_dataset_v2.py` → `dataset_manifest.json`, `DATASET_CARD.json`.
- **Diagnostic tokenizer-only (not student lock-in):** `check_open_dataset_tokenizer_only.py` → `diagnostic_tokenizer_audit.json` (23 edits ≤ 16384 tokens; max 13257).

### 6. Revocation and honesty

- **Revocation record:** `revocation_manifest.json` (v1 sample revoked; full v1 revocation history referenced in provenance).
- **Admission ledger:** `acceptance_ledger.json`.

## Conclusions encoded in the snapshot

- Generic schema, bounded behavior/preservation, input/leakage audit, and repo-group split checks passed for the 23 admitted edits.
- `target_student_selected`: false; `formal_training_ready`: false.
- `full_upstream_test_suite_run`: false; `terminal_bench_transfer_verified`: false; `collection_complete`: false.
- Research1 formal campaign (V14) remains at qualified edit 0 — see [`records/2026-11-01/status.json`](../records/2026-11-01/status.json).

## Limits (do not over-claim)

- Public issue/change Plans are **editorial** demonstrations; compiler patches and gold tests are evidence, not policy inputs.
- Probes use recorded dependency/IO fixtures — not full application environments.
- Published solver pass/fail must not be relabeled as G/D/U or formal MC.
- Scripts in `scripts/open_data/` are an **archive** of the integration workspace workflow; paths inside them were not rewritten for universal clone-and-run reproduction in this task.

## Related docs

- [`docs/code_map.md`](code_map.md) — research1 `data` CLI (separate from this open auxiliary import).
- [`docs/data_flow_and_sources.md`](data_flow_and_sources.md) — external source survey.
- [`docs/reports_catalog.md`](reports_catalog.md) — public records vs server-only pointers.

## Additional attribution for raw records

The retained raw corpus also contains two OpenHands records and one mem0 record outside the accepted samples. Their original parent-revision license texts and source URLs are preserved in [raw record license inventory](../records/2026-11-02/raw_record_licenses/index.json). The accepted 23-sample repository count remains four. Attribution for accepted source snapshots is in [NOTICE](../data/open_auxiliary/v2/NOTICE.md).
