# NOTICE — third-party and compilation attribution

This tree ships **research auxiliary dataset v2** (23 bounded planning edits; auxiliary only, not formal autonomous SFT). It includes redistributed upstream source excerpts under `portable_sources/`, `refs/`, and related paths, plus compilation metadata derived from Nebius SWE-rebench Hugging Face releases.

## 1. Dataset compilation (SWE-rebench family)

Curated records, editorial Plans, and compilation metadata are derived from Nebius SWE-rebench datasets. **Compilation license: CC-BY-4.0** (see `source_cards.json`).

| Dataset | Revision (pin) | Source |
| --- | --- | --- |
| nebius/SWE-rebench | `89cdfbab4ab1bd8f5a658bb212d1b63624f4f881` | https://huggingface.co/datasets/nebius/SWE-rebench/tree/89cdfbab4ab1bd8f5a658bb212d1b63624f4f881 |
| nebius/SWE-rebench-V2 | `10483de0f50fe5da545942705a76c6150171af7f` | https://huggingface.co/datasets/nebius/SWE-rebench-V2/tree/10483de0f50fe5da545942705a76c6150171af7f |
| nebius/SWE-rebench-V2-PRs | `fbf0ecf50f268d5344149e2f0097db6bede83737` | https://huggingface.co/datasets/nebius/SWE-rebench-V2-PRs/tree/fbf0ecf50f268d5344149e2f0097db6bede83737 |

Immutable release archive (not stored in this git tree): `research1_open_auxiliary_v2.zip` SHA-256 `d7de1479d022de7fdfc4f6618434b0231f7b82ac428aca53ed84cc91591cc40b`. See `repo_packaging_manifest.json`.

## 2. Upstream application source (four parent repositories)

Redistributed code fragments retain upstream licenses at the pinned `source_base_commit` recorded in `provenance.json` and `source_records/*.json`. They are **not** licensed as CC-BY-4.0.

Verbatim upstream `LICENSE` files (one file per distinct license text SHA-256) live in `third_party_licenses/`:

| Upstream repo | License (bundled open-source paths) | Canonical file(s) in `third_party_licenses/` |
| --- | --- | --- |
| BerriAI/litellm | MIT | `sha256_b170d6bf8e8835dd357e011681db028f4d51e2fb0ea892058f56e01fb39b8273__BerriAI_litellm__LICENSE` |
| crewAIInc/crewAI | MIT (two text variants at pinned commits) | `sha256_28868731…__crewAIInc_crewAI__LICENSE`, `sha256_d9b9bdb8…__crewAIInc_crewAI__LICENSE` |
| huggingface/smolagents | Apache-2.0 | `sha256_c71d239d…__huggingface_smolagents__LICENSE` |
| stanfordnlp/dspy | MIT | `sha256_fc84421f…__stanfordnlp_dspy__LICENSE` |

Commit-to-sample mapping and fetch URLs: `third_party_licenses/LICENSE_BY_SHA256.json`, `third_party_licenses/_license_fetch_log.json`. Resolved summary (non-legal): `resolved_license_inventory.json`.

### LiteLLM scope

Upstream root `LICENSE` states that content under `enterprise/` is under a separate commercial license. **`portable_sources/berriai__litellm-*` trees in this release contain no `enterprise/` path segments**; bundled paths are open-source SDK and documentation excerpts. README snippets may mention Enterprise features as documentation only.

### Apache-2.0 smolagents excerpts

Files under `portable_sources/huggingface__smolagents-*` include per-file Apache-2.0 headers where present upstream. Ship and preserve those headers together with the full upstream `LICENSE` file above.

## 3. Layout

```
data/open_auxiliary/v2/
  NOTICE.md
  resolved_license_inventory.json
  third_party_licenses/     # five sha256-named LICENSE files + indexes
  source_cards.json
  provenance.json
  source_records/
```

## 4. Attribution minimum

- Retain `source_cards.json`, `provenance.json`, and per-sample `source_records/*.json`.
- Do not replace upstream code licenses with CC-BY-4.0.
- For the CC-BY-4.0 compilation, credit Nebius SWE-rebench dataset releases at the revisions in section 1.
