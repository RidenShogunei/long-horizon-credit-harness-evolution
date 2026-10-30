"""Call ModularRSI Harbor evaluate / official phase0 evolve; inject proxy + provider."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research1.config import REPO_ROOT


DEFAULT_PROXY = "http://172.17.0.1:7890"
# LLM API hosts must bypass the local clash proxy: Tongji returns
# HTTP 403「当前网络环境不支持」when the request egresses via :7890.
DEFAULT_NO_PROXY = (
    "localhost,127.0.0.1,::1,api.deepseek.com,agi.tongji.edu.cn"
)
LLM_NO_PROXY_HOSTS = ("api.deepseek.com", "agi.tongji.edu.cn")


@dataclass
class ProviderEnv:
    provider_id: str
    mode: str
    api_base: str
    model: str
    api_key: str
    raw: dict[str, str]


def with_llm_provider_env(mode: str | None = None) -> ProviderEnv:
    """Resolve provider via scripts/llm_provider.py (same as with_llm_provider.sh)."""
    cmd = [
        "python3",
        str(REPO_ROOT / "scripts/llm_provider.py"),
        "export-env",
        "--with-meta",
    ]
    if mode:
        cmd.extend(["--mode", mode])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"llm_provider export-env failed: {proc.stderr}")
    exports: dict[str, str] = {}
    meta_provider = ""
    meta_mode = ""
    meta_model = ""
    meta_base = ""
    for line in proc.stdout.splitlines():
        if line.startswith("# provider="):
            # # provider=X mode=Y model=Z base=W
            parts = dict(
                p.split("=", 1) for p in line[2:].split() if "=" in p
            )
            meta_provider = parts.get("provider", "")
            meta_mode = parts.get("mode", "")
            meta_model = parts.get("model", "")
            meta_base = parts.get("base", "")
            continue
        if line.startswith("export "):
            body = line[len("export ") :]
            k, _, v = body.partition("=")
            exports[k] = shlex.split(v)[0] if v else ""
    return ProviderEnv(
        provider_id=meta_provider or exports.get("LLM_PROVIDER_ID", ""),
        mode=meta_mode or exports.get("LLM_PROVIDER_MODE", ""),
        api_base=meta_base or exports.get("API_BASE", ""),
        model=meta_model or exports.get("MODEL") or exports.get("HARBOR_MODEL", ""),
        api_key=exports.get("HARBOR_EVO_API_KEY") or exports.get("OPENAI_API_KEY", ""),
        raw=exports,
    )


def proxy_url() -> str:
    return os.environ.get("PROXY_URL") or DEFAULT_PROXY


def merge_no_proxy(existing: str | None = None) -> str:
    """Ensure LLM API hosts are never sent through the download proxy."""
    parts: list[str] = []
    seen: set[str] = set()
    for chunk in (existing or "", DEFAULT_NO_PROXY):
        for item in chunk.split(","):
            host = item.strip()
            if not host:
                continue
            key = host.lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(host)
    for host in LLM_NO_PROXY_HOSTS:
        if host.lower() not in seen:
            parts.append(host)
            seen.add(host.lower())
    return ",".join(parts)


def apply_provider_to_environ(prov: ProviderEnv) -> None:
    for k, v in prov.raw.items():
        os.environ[k] = v
    # Host-side proxy for HF / downloads; agent/verifier get --ve/--ae separately.
    # LLM bases (DeepSeek official + Tongji) must stay on NO_PROXY.
    px = proxy_url()
    os.environ.setdefault("HTTP_PROXY", px)
    os.environ.setdefault("HTTPS_PROXY", px)
    os.environ.setdefault("http_proxy", px)
    os.environ.setdefault("https_proxy", px)
    merged = merge_no_proxy(os.environ.get("NO_PROXY") or os.environ.get("no_proxy"))
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged


def harbor_python(modularrsi_root: Path) -> Path:
    venv_py = modularrsi_root / ".venv/bin/python"
    if venv_py.is_file():
        return venv_py
    raise FileNotFoundError(f"ModularRSI venv python missing: {venv_py}")


def check_harbor_import(modularrsi_root: Path) -> Path:
    py = harbor_python(modularrsi_root)
    code = (
        "from pathlib import Path; import harbor, sys; "
        f"expected=(Path({str(modularrsi_root)!r})/'src/harbor/__init__.py').resolve(); "
        "actual=Path(harbor.__file__).resolve(); "
        "assert actual==expected, (actual, expected); print(actual)"
    )
    proc = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"harbor import check failed: {proc.stderr or proc.stdout}")
    return Path(proc.stdout.strip())


def run_harbor_task(
    *,
    modularrsi_root: Path,
    modules_root: Path,
    task_name: str,
    output_dir: Path,
    model: str,
    api_base: str,
    temperature: float,
    dataset: str = "terminal-bench@2.0",
    dataset_path: Path | None = None,
    locked_module_type: str = "tools",
    composer_name: str = "static",
    max_turns: int = 200,
    environment: str = "docker",
    attempts: int = 1,
    timeout_sec: int | None = 14400,
) -> subprocess.CompletedProcess[str]:
    """Run one Harbor evaluate via ModularRSI python (not rewriting the executor)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    py = harbor_python(modularrsi_root)
    px = proxy_url()
    no_proxy = merge_no_proxy(os.environ.get("NO_PROXY") or os.environ.get("no_proxy"))
    model_info = os.environ.get(
        "HARBOR_MODEL_INFO",
        '{"max_input_tokens":176000,"max_output_tokens":16000,'
        '"input_cost_per_token":0.0,"output_cost_per_token":0.0}',
    )
    cmd = [
        str(py),
        "-m",
        "harbor.cli.main",
        "run",
        "--yes",
        "-a",
        "terminus-2-modular",
        "-m",
        model,
        "-e",
        environment,
        "-n",
        "1",
        "-k",
        str(attempts),
        "-o",
        str(output_dir),
        "--ak",
        f"modules_root={modules_root}",
        "--ak",
        f"api_base={api_base}",
        "--ak",
        f"max_turns={max_turns}",
        "--ak",
        f"model_info={model_info}",
        "--ak",
        f"temperature={temperature}",
        "--ak",
        f"locked_module_type={locked_module_type}",
        "--ak",
        "composer_scope=all",
        "--ak",
        f"composer_name={composer_name}",
    ]
    # Prefer local dataset path when provided (ModularRSI_2000_Instances/tb)
    if dataset_path is not None and (dataset_path / "tasks" / task_name).is_dir():
        cmd.extend(["-p", str(dataset_path / "tasks" / task_name)])
    else:
        cmd.extend(["-d", dataset, "-i", task_name])

    for flag in ("--ve", "--ae"):
        for k, v in (
            ("HTTP_PROXY", px),
            ("HTTPS_PROXY", px),
            ("http_proxy", px),
            ("https_proxy", px),
            ("NO_PROXY", no_proxy),
            ("no_proxy", no_proxy),
        ):
            cmd.extend([flag, f"{k}={v}"])

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Host Harbor/LiteLLM inherits this env; keep LLM hosts off the clash proxy.
    env["NO_PROXY"] = no_proxy
    env["no_proxy"] = no_proxy
    return subprocess.run(
        cmd,
        cwd=str(modularrsi_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


def harbor_proxy_python() -> Path:
    """Python whose directory's `harbor` shim injects container download proxy.

    evolve.sh prepends dirname(PYTHON_BIN) to PATH. The shim is not upstream.
    """
    bindir = REPO_ROOT / "scripts" / "harbor_proxy_path"
    py = bindir / "python"
    harbor = bindir / "harbor"
    if not harbor.is_file() or not py.exists():
        raise FileNotFoundError(
            f"harbor proxy PATH missing under {bindir} (need python symlink and harbor)"
        )
    return py


def run_official_evolve_smoke(
    *,
    modularrsi_root: Path,
    support_dataset_dir: Path,
    locked_module: str,
    archive_root: Path,
    model: str,
    api_base: str,
    profile: str = "smoke",
    max_tasks: int | None = None,
    task_seed: int = 0,
    dry_run: bool = False,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke upstream scripts/evolve.sh with PROFILE=smoke (official Phase0)."""
    archive_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "MODEL": model,
            "API_BASE": api_base,
            "SUPPORT_DATASET_DIR": str(support_dataset_dir),
            "SUPPORT_SPLIT": "train",
            "LOCKED_MODULE": locked_module,
            "PROFILE": profile,
            "ARCHIVE_ROOT": str(archive_root),
            "ENVIRONMENT": "docker",
            "DRY_RUN": "true" if dry_run else "false",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TASK_SEED": str(task_seed),
            # evolve.sh puts dirname(PYTHON_BIN) first on PATH so phase0's
            # `harbor` is the container-proxy shim, not a second algorithm.
            "PYTHON_BIN": str(harbor_proxy_python()),
        }
    )
    if max_tasks is not None:
        env["MAX_TASKS"] = str(max_tasks)
    # Inject proxy into host env; phase0/harbor may inherit for downloads.
    px = proxy_url()
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env[k] = px
    merged = merge_no_proxy(env.get("NO_PROXY") or env.get("no_proxy"))
    env["NO_PROXY"] = merged
    env["no_proxy"] = merged
    if extra_env:
        env.update(extra_env)
    script = modularrsi_root / "scripts/evolve.sh"
    # Log command
    log_path = archive_root / "evolve_invoke.json"
    log_path.write_text(
        json.dumps(
            {
                "script": str(script),
                "PROFILE": profile,
                "LOCKED_MODULE": locked_module,
                "SUPPORT_DATASET_DIR": str(support_dataset_dir),
                "ARCHIVE_ROOT": str(archive_root),
                "MODEL": model,
                "API_BASE": api_base,
                "TASK_SEED": task_seed,
                "NO_PROXY": env.get("NO_PROXY"),
                "PYTHON_BIN": env.get("PYTHON_BIN"),
                "dry_run": dry_run,
                "started_unix": time.time(),
            },
            indent=2,
        )
        + "\n"
    )
    return subprocess.run(
        ["bash", str(script)],
        cwd=str(modularrsi_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=None if dry_run else 86400,
    )


def write_local_adaptations_note(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Local adaptations vs upstream ModularRSI (A13)

## What we call upstream as-is
- `bash scripts/evolve.sh` with `PROFILE=smoke` → `python -m harbor.agents.terminus_2_modular.self_evo.phase0`
- Algorithmic Phase0 gates unchanged; smoke only scales workload (max_tasks=2, epochs=1, …)
- Solver temperature for P2A follows upstream Phase0 constant `0.0`

## Local adaptations (do not claim as upstream)
- Provider selection via this repo's `scripts/with_llm_provider.sh` / `llm_provider.py`
- Mandatory host proxy `http://172.17.0.1:7890`, with `NO_PROXY` including `agi.tongji.edu.cn` so Tongji is not sent through clash
- Container download proxy injected by `scripts/harbor_proxy_path/harbor` (`--ae`/`--ve`) because this Harbor pin's phase0 runner does not pass those flags. `PYTHON_BIN` points at that directory so `evolve.sh` puts the shim first on `PATH`
- `TASK_SEED` is forwarded. Upstream slices the train manifest to `max_tasks` and only then shuffles, so `PROFILE=smoke` always uses the first two train tasks (`browser-playable-leaderboard`, `build-ffmpeg-legacy`). No seed removes the heavy image task
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
"""
    )


def save_process_logs(output_dir: Path, proc: subprocess.CompletedProcess[str], name: str = "console") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.stdout.log").write_text(proc.stdout or "")
    (output_dir / f"{name}.stderr.log").write_text(proc.stderr or "")
    (output_dir / f"{name}.exitcode").write_text(str(proc.returncode))
