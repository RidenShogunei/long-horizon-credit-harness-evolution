# Open auxiliary v2 — repository organization record (2026-11-02)

This document records how the root-approved auxiliary dataset v2 snapshot and related workflow scripts were placed in this repository. It is not a reproduction guide.

## What was imported

| Component | Location |
| --- | --- |
| Sealed dataset payload (289 manifest-hashed paths) | `data/open_auxiliary/v2/` |
| Sealed manifest archive | `data/open_auxiliary/v2/archive/sealed_dataset_manifest_release_v2.json` |
| Repo packaging metadata | `data/open_auxiliary/v2/repo_packaging_manifest.json` |
| Third-party LICENSE texts (5 canonical sha256-named files) | `data/open_auxiliary/v2/third_party_licenses/` |
| Compilation NOTICE + resolved license inventory | `data/open_auxiliary/v2/NOTICE.md`, `resolved_license_inventory.json` |
| Workflow script archive | `scripts/open_data/` (28 Python copies; index in `INDEX.md`) |

Source bytes: extracted `open_dataset_integration_v2/release_v2`. Immutable zip reference (not checked in): `research1_open_auxiliary_v2.zip` SHA-256 `d7de1479d022de7fdfc4f6618434b0231f7b82ac428aca53ed84cc91591cc40b`.

## Integrity (organization checks only)

- All **289** paths in `dataset_manifest.json` `files{}` match sealed SHA-256 under `data/open_auxiliary/v2/`.
- Sample counts: **23** auxiliary edit (**20** train / **3** validation), keep **0**; raw corpus **96** lines in `source_record_corpus/combined_source_records_96.jsonl`.
- **301** files under `data/open_auxiliary/v2/` total (**289** sealed payload + **12** packaging/attribution files not listed in `files{}`; see `repo_packaging_manifest.json`).
- `source_cards.json` unchanged, including any unknown license markers; `resolved_license_inventory.json` adds non-legal LiteLLM MIT / non-`enterprise/` path metadata only.

## Exclusions (not tracked)

- Duplicate `research1_open_auxiliary_v2.zip`
- ~429MB parquet download caches and full HuggingFace dataset mirrors
- `__pycache__`, `.pyc`, nested `.git`, `.pytest_cache`, temp probe scratch
- Model weight checkpoints and Cursor session stdout/thinking logs
- Credentials and API keys

## Cleanup actions

- Deduplicated repeated `.gitignore` blocks (C28 duplicate section merged).
- Copied dataset with robocopy filters excluding `__pycache__` / `.pyc` on import.
- Did **not** recursively delete or rewrite the original integration workspace.

## Explicitly retained evidence

- Non-admitted, pending, failed verification, and reconstruction-failed raw ledger rows.
- Rejected curator drafts, revocation records, and negative outcomes under `portable_proofs/revoked/`.
- Root verification JSON, portable proof fixtures, and pinned `portable_sources/` / `refs/` trees.
- Failed or partial teacher/curation artifacts embedded in the sealed payload.

## Historical research artifacts

- **119** tracked files under `runs/` remain versioned historical evidence; `.gitignore` suppresses new runtime noise but does not retroactively untrack committed run summaries.
- Research1 formal V14 campaign state remains in `records/2026-11-01/status.json` (qualified edit **0**, formal SFT **0**, MC **0**).
- Main research pipeline under `src/research1/` unchanged.

## Frozen metadata notes

- Some sealed JSON (e.g. `diagnostic_tokenizer_audit.json`) may reference non-portable server paths inside historical fields; public docs use repository-relative paths.
- Substrings such as `api_key` appear in pinned upstream source excerpts (LiteLLM/CrewAI), not as live secrets.
