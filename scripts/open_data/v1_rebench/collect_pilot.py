#!/usr/bin/env python3
"""
C49 pilot: nebius/SWE-rebench harness-repo rows only.
Bounded HF dataset-server /filter fetch (<=100MB), patch screen, GitHub source + git apply --check.
Run from repo root or this directory: python collect_pilot.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
HF_DATASET = "nebius/SWE-rebench"
HF_CARD_LICENSE = "CC-BY-4.0"
MAX_FETCH_BYTES = 100 * 1024 * 1024
MAX_RAW_ROWS = 100
MAX_USABLE = 20
USER_AGENT = "open_dataset_integration_v1/rebench-pilot"

PRIMARY_ALLOWLIST = [
    "Aider-AI/aider",
    "SWE-agent/SWE-agent",
    "All-Hands-AI/OpenHands",
    "OpenDevin/OpenDevin",
    "langchain-ai/langchain",
    "langchain-ai/langgraph",
    "microsoft/autogen",
    "huggingface/smolagents",
    "crewAIInc/crewAI",
    "run-llama/llama_index",
    "ag2ai/ag2",
]

HARNESS_REPO_KEYWORDS = (
    "aider",
    "swe-agent",
    "openhands",
    "opendevin",
    "langchain",
    "langgraph",
    "autogen",
    "smolagent",
    "crewai",
    "llama_index",
    "llama-index",
    "ag2",
    "agent",
    "openhands",
)

HARNESS_PATH_POSITIVE = re.compile(
    r"(^|/)(tools?|agent|agents|harness|runtime|orchestr|observ|context|"
    r"verification|verify|executor|controller|loop|action|environment|envs?|"
    r"sandbox|codeact|swerex|commands?)(/|$|\.)",
    re.I,
)

EXCLUDE_PATH = re.compile(
    r"(^|/)(docs?|documentation|examples?|benchmarks?|scripts?/benchmark|"
    r"frontend|ui/|\.github/|assets/|images?/|notebooks?)(/|$)|"
    r"\.(md|rst|png|jpg|svg|ico|yaml|yml|toml|json|lock|ini|cfg)$|"
    r"(^|/)(tests?|testing|__tests__|spec)(/|$)|test_.*\.py$|.*_test\.py$",
    re.I,
)

CONFIG_ONLY = re.compile(r"(^|/)(config|configs|settings|\.vscode|\.idea)(/|$)", re.I)

DIFF_FILE = re.compile(r"^\+\+\+ b/(.*)$")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


class ByteBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def add(self, n: int) -> None:
        self.used += n
        if self.used > self.limit:
            raise RuntimeError(f"Download budget exceeded: {self.used} > {self.limit}")


BUDGET = ByteBudget(MAX_FETCH_BYTES)


def http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(BUDGET.limit - BUDGET.used + 1)
    BUDGET.add(len(data))
    return data


def http_get_json(url: str) -> Any:
    raw = http_get(url)
    return json.loads(raw.decode("utf-8"))


def save_hf_response(name: str, url: str, payload: bytes | str) -> dict[str, Any]:
    d = ROOT / "artifacts" / "hf_responses"
    d.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    h = sha256_bytes(payload)
    path = d / f"{name}_{h[:12]}.json"
    path.write_bytes(payload)
    return {"url": url, "path": str(path.relative_to(ROOT)), "sha256": h, "bytes": len(payload)}


def resolve_revision() -> str:
    url = f"https://huggingface.co/api/datasets/{HF_DATASET}/revision/main"
    revs = http_get_json(url)
    if isinstance(revs, dict) and isinstance(revs.get('sha'), str):
        return revs['sha']
    if isinstance(revs, list) and revs:
        return revs[0]["sha"]
    raise RuntimeError("Could not resolve dataset revision")


def dataset_info() -> dict[str, Any]:
    url = f"https://datasets-server.huggingface.co/info?dataset={HF_DATASET}"
    return http_get_json(url)


def pick_config_split(info: dict[str, Any]) -> tuple[str, str]:
    di = info.get("dataset_info") or info.get("dataset_infos") or {}
    if not di:
        return "default", "test"
    cfg = next(iter(di.keys()))
    splits = di[cfg].get("splits") or {}
    split = "test" if "test" in splits else next(iter(splits.keys()))
    return cfg, split


def filter_rows(
    config: str, split: str, where: str, offset: int = 0, length: int = 100
) -> dict[str, Any]:
    q = urllib.parse.urlencode(
        {
            "dataset": HF_DATASET,
            "config": config,
            "split": split,
            "where": where,
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/filter?{q}"
    return http_get_json(url)


def statistics_column(config: str, column: str) -> dict[str, Any]:
    q = urllib.parse.urlencode({"dataset": HF_DATASET, "config": config, "column": column})
    url = f"https://datasets-server.huggingface.co/statistics?{q}"
    try:
        return http_get_json(url)
    except urllib.error.HTTPError:
        return {}


def sql_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def build_in_clause(repos: list[str]) -> str:
    return '(' + ' OR '.join('"repo"=' + sql_quote(r) for r in repos) + ')'


def row_id(row: dict[str, Any], idx: int) -> str:
    for k in ("instance_id", "id", "task_id"):
        if row.get(k):
            return str(row[k])
    repo = row.get("repo", "?")
    commit = row.get("base_commit", "?")[:8]
    return f"{repo}@{commit}#{idx}"


def parse_patch_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for line in (patch or "").splitlines():
        m = DIFF_FILE.match(line)
        if m:
            p = m.group(1).strip()
            if p != "/dev/null":
                paths.append(p)
    return paths


def classify_patch(paths: list[str]) -> tuple[str, list[str]]:
    if not paths:
        return "empty_patch", paths
    if all(EXCLUDE_PATH.search(p) for p in paths):
        return "all_excluded_paths", paths
    if all(CONFIG_ONLY.search(p) for p in paths):
        return "config_only", paths
    harness_paths = [
        p
        for p in paths
        if HARNESS_PATH_POSITIVE.search(p) and not EXCLUDE_PATH.search(p)
    ]
    if harness_paths:
        return "harness_relevant", paths
    code_paths = [p for p in paths if p.endswith((".py", ".rs", ".go", ".ts", ".tsx", ".js"))]
    if not code_paths:
        return "non_code", paths
    # Harness repo task: retain if any non-test/non-doc code path
    productive = [
        p
        for p in code_paths
        if not EXCLUDE_PATH.search(p) and not CONFIG_ONLY.search(p)
    ]
    if productive:
        return "code_change", paths
    return "excluded_category", paths


def issue_key(rid: str, repo: str) -> str:
    m = re.search(r"(\d+)\s*$", rid.replace("_", "-"))
    num = m.group(1) if m else rid
    return f"{repo}#{num}"


def repo_train_dev(repos: list[str]) -> dict[str, str]:
    unique = sorted(set(repos))
    mid = len(unique) // 2
    train = set(unique[:mid] if mid else unique[:1])
    return {r: ("train" if r in train else "dev") for r in unique}


def github_raw(repo: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def fetch_source_file(repo: str, commit: str, path: str) -> bytes | None:
    url = github_raw(repo, commit, path)
    try:
        return http_get(url, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def try_ast_parse(path: str, content: bytes) -> dict[str, Any]:
    if not path.endswith(".py"):
        return {"parsed": None, "skipped": "not_python"}
    try:
        text = content.decode("utf-8")
        ast.parse(text)
        return {"parsed": True}
    except SyntaxError as ex:
        return {"parsed": False, "error": str(ex)}
    except UnicodeDecodeError as ex:
        return {"parsed": False, "error": str(ex)}


def git_apply_check(repo: str, commit: str, patch: str) -> dict[str, Any]:
    if not shutil.which("git"):
        return {"ok": False, "skipped": "git_not_found"}
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        r = subprocess.run(
            ["git", "init"],
            cwd=td_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            return {"ok": False, "error": r.stderr}
        subprocess.run(
            ["git", "remote", "add", "origin", f"https://github.com/{repo}.git"],
            cwd=td_path,
            capture_output=True,
            check=False,
        )
        fetch = subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", commit],
            cwd=td_path,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if fetch.returncode != 0:
            return {"ok": False, "error": fetch.stderr[:2000]}
        co = subprocess.run(
            ["git", "checkout", "FETCH_HEAD"],
            cwd=td_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if co.returncode != 0:
            return {"ok": False, "error": co.stderr[:2000]}
        patch_file = td_path / "candidate.patch"
        patch_file.write_text(patch, encoding="utf-8")
        chk = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", str(patch_file)],
            cwd=td_path,
            capture_output=True,
            text=True,
            check=False,
        )
        strict = subprocess.run(
            ["git", "apply", "--check", "--strict", "--whitespace=nowarn", str(patch_file)],
            cwd=td_path,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "ok": chk.returncode == 0,
            "strict_ok": strict.returncode == 0,
            "stderr": (chk.stderr or chk.stdout)[:1500],
            "strict_stderr": (strict.stderr or strict.stdout)[:1500],
        }


def discover_repos_from_stats(config: str) -> list[str]:
    stats = statistics_column(config, "repo")
    found: list[str] = []
    for blob in (stats.get("statistics") or stats.get("column_statistics") or []):
        if isinstance(blob, dict):
            for item in blob.get("values", []) or blob.get("top_values", []) or []:
                val = item.get("value") if isinstance(item, dict) else item
                if not val or not isinstance(val, str):
                    continue
                low = val.lower()
                if any(k in low for k in HARNESS_REPO_KEYWORDS):
                    found.append(val)
    # dedupe preserve order
    seen: set[str] = set()
    out: list[str] = []
    for r in found:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def extract_outcomes(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {"raw": meta}
    keys = (
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "FAIL_TO_FAIL",
        "PASS_TO_FAIL",
        "resolved",
        "license_name",
    )
    return {k: row[k] if k in row else meta[k] for k in keys if k in row or k in meta}


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "dataset": HF_DATASET,
        "card_license": HF_CARD_LICENSE,
        "compilation_note": "Per-row source license in meta.license_name; compilation attribution CC-BY-4.0",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_limits": {"max_fetch_bytes": MAX_FETCH_BYTES, "max_rows": MAX_RAW_ROWS, "max_usable": MAX_USABLE},
        "hf_responses": [],
        "revision": None,
        "config": None,
        "split": None,
        "allowlist_primary": PRIMARY_ALLOWLIST,
        "allowlist_effective": [],
    }
    exclusion_counts: Counter[str] = Counter()

    try:
        revision = resolve_revision()
    except Exception as ex:
        print(f"FATAL revision: {ex}", file=sys.stderr)
        return 2
    manifest["revision"] = revision

    try:
        info = dataset_info()
    except Exception as ex:
        print(f"FATAL info: {ex}", file=sys.stderr)
        return 2
    raw_info = json.dumps(info).encode("utf-8")
    manifest["hf_responses"].append(
        save_hf_response("info", f"https://datasets-server.huggingface.co/info?dataset={HF_DATASET}", raw_info)
    )
    config, split = pick_config_split(info)
    manifest["config"] = config
    manifest["split"] = split

    collected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    repos_tried: list[str] = []

    def unwrap_hf_row(item: Any) -> dict[str, Any]:
        if isinstance(item, dict) and "row" in item and isinstance(item["row"], dict):
            out = dict(item["row"])
            if "row_idx" in item:
                out["_hf_row_idx"] = item["row_idx"]
            return out
        return dict(item) if isinstance(item, dict) else {}

    def ingest_filter_response(resp: dict[str, Any], where: str, url_meta: str) -> None:
        payload = json.dumps(resp).encode("utf-8")
        manifest["hf_responses"].append(save_hf_response("filter", url_meta, payload))
        for i, item in enumerate(resp.get("rows") or []):
            row = unwrap_hf_row(item)
            rid = row_id(row, i)
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            collected.append(
                {
                    "source_row_id": rid,
                    "dataset_split": split,
                    "filter_where": where,
                    **row,
                }
            )

    # Primary allowlist
    where_primary = build_in_clause(PRIMARY_ALLOWLIST)
    repos_tried.extend(PRIMARY_ALLOWLIST)
    try:
        resp = filter_rows(config, split, where_primary, 0, MAX_RAW_ROWS)
        ingest_filter_response(
            resp,
            where_primary,
            f"filter primary allowlist length={MAX_RAW_ROWS}",
        )
    except Exception as ex:
        exclusion_counts["filter_primary_error"] += 1
        print(f"warn primary filter: {ex}", file=sys.stderr)

    # Broaden from statistics if needed
    effective = list(PRIMARY_ALLOWLIST)
    if len(collected) < 10:
        extra = discover_repos_from_stats(config)
        manifest["discovered_harness_like_repos"] = extra
        new_repos = [r for r in extra if r not in effective]
        if new_repos:
            effective.extend(new_repos)
            where_extra = build_in_clause(new_repos[:40])
            try:
                resp2 = filter_rows(config, split, where_extra, 0, MAX_RAW_ROWS - len(collected))
                ingest_filter_response(resp2, where_extra, "filter discovered repos")
            except Exception as ex:
                exclusion_counts["filter_discover_error"] += 1
                print(f"warn discover filter: {ex}", file=sys.stderr)

    manifest["allowlist_effective"] = effective
    collected = collected[:MAX_RAW_ROWS]

    # Per-row repo verification
    verified_rows: list[dict[str, Any]] = []
    for row in collected:
        repo = row.get("repo") or ""
        if repo not in effective and not any(repo == p for p in PRIMARY_ALLOWLIST):
            exclusion_counts["repo_not_in_effective_allowlist"] += 1
            continue
        verified_rows.append(row)

    raw_path = ROOT / "raw_selected_rows.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for row in verified_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    partition = repo_train_dev([r.get("repo", "") for r in verified_rows])

    normalized_path = ROOT / "normalized_records.jsonl"
    screened: list[dict[str, Any]] = []
    with normalized_path.open("w", encoding="utf-8") as f:
        for row in verified_rows:
            paths = parse_patch_paths(row.get("patch") or "")
            screen, _ = classify_patch(paths)
            exclusion_counts[f"screen_{screen}"] += 1
            outcomes = extract_outcomes(row)
            rec = {
                "source_row_id": row["source_row_id"],
                "lineage": {
                    "dataset": HF_DATASET,
                    "revision": revision,
                    "config": config,
                    "split": row.get("dataset_split", split),
                    "filter_where": row.get("filter_where"),
                    "hf_response_hashes": [h["sha256"] for h in manifest["hf_responses"]],
                },
                "repo": row.get("repo"),
                "base_commit": row.get("base_commit"),
                "problem_statement": row.get("problem_statement"),
                "patch": row.get("patch"),
                "patch_paths": paths,
                "patch_screen": screen,
                "meta_license_name": row.get('license_name') or ((row.get('meta') or {}).get('license_name') if isinstance(row.get('meta'), dict) else None),
                "repo_partition": partition.get(row.get("repo", ""), "dev"),
                "supervised_plan_fields_auxiliary": False,
                "curator_original_plan": False,
                "source_published_task_actual": True,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if screen in ("harness_relevant", "code_change"):
                screened.append({**rec, "outcomes_separate": outcomes})

    # Diversity: one per issue_key, prefer harness_relevant
    screened.sort(key=lambda r: (0 if r["patch_screen"] == "harness_relevant" else 1, r["source_row_id"]))
    chosen: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    for rec in screened:
        ik = issue_key(rec["source_row_id"], rec.get("repo") or "")
        if ik in used_keys:
            exclusion_counts["duplicate_issue"] += 1
            continue
        used_keys.add(ik)
        chosen.append(rec)
        if len(chosen) >= MAX_USABLE:
            break
    exclusion_counts["screened_not_selected"] += max(0, len(screened) - len(chosen))

    pre_dir = ROOT / "predecision_input"
    post_dir = ROOT / "postdecision_outcomes"
    src_dir = ROOT / "sources"
    pre_dir.mkdir(exist_ok=True)
    post_dir.mkdir(exist_ok=True)
    src_dir.mkdir(exist_ok=True)

    usable_path = ROOT / "usable_candidates.jsonl"
    with usable_path.open("w", encoding="utf-8") as uf:
        for rec in chosen:
            repo = rec["repo"]
            commit = rec["base_commit"]
            paths = [p for p in rec["patch_paths"] if p.endswith(".py") and not EXCLUDE_PATH.search(p)]
            if not paths:
                paths = [p for p in rec["patch_paths"] if not EXCLUDE_PATH.search(p)][:5]
            file_bundle: dict[str, Any] = {}
            for p in paths[:8]:
                content = fetch_source_file(repo, commit, p)
                if content is None:
                    exclusion_counts["source_fetch_404"] += 1
                    continue
                rel = ROOT / "sources" / rec["source_row_id"].replace("/", "_") / p
                rel.parent.mkdir(parents=True, exist_ok=True)
                rel_full = ROOT / rel
                rel_full.parent.mkdir(parents=True, exist_ok=True)
                rel_full.write_bytes(content)
                file_bundle[p] = {
                    "path": str(rel.relative_to(ROOT)),
                    "sha256": sha256_bytes(content),
                    "bytes": len(content),
                    "ast": try_ast_parse(p, content),
                }

            apply_result = git_apply_check(repo, commit, rec.get("patch") or "")

            pre = {
                "source_row_id": rec["source_row_id"],
                "repo": repo,
                "base_commit": commit,
                "repo_partition": rec["repo_partition"],
                "problem_statement": rec.get("problem_statement"),
                "decision_time_source_files": file_bundle,
                "note": "No test_patch, hidden tests, or gold repair hints included.",
            }
            post = {
                "source_row_id": rec["source_row_id"],
                "patch": rec.get("patch"),
                "patch_apply_check": apply_result,
                "outcomes_from_dataset_meta": rec.get("outcomes_separate") or extract_outcomes(
                    next(r for r in verified_rows if r["source_row_id"] == rec["source_row_id"])
                ),
                "outcomes_not_planning_evidence": True,
            }
            (pre_dir / f"{rec['source_row_id'].replace('/', '_')}.json").write_text(
                json.dumps(pre, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (post_dir / f"{rec['source_row_id'].replace('/', '_')}.json").write_text(
                json.dumps(post, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            uf.write(
                json.dumps(
                    {
                        **rec,
                        "decision_time_source_files": file_bundle,
                        "patch_apply_check": apply_result,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest["byte_budget_used"] = BUDGET.used
    manifest["counts"] = {
        "raw_selected": len(verified_rows),
        "normalized": len(verified_rows),
        "screened_eligible": len(screened),
        "usable_candidates": len(chosen),
    }
    (ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (ROOT / "exclusion_counts.json").write_text(
        json.dumps(dict(exclusion_counts), indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest["counts"], indent=2))
    print(f"bytes_used={BUDGET.used} revision={revision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
