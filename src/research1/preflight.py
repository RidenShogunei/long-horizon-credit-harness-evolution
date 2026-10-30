"""Environment blockers: proxy, provider, disk, harbor, dataset, H0 hashes."""

from __future__ import annotations

import json
import socket
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from research1 import EXIT_FAIL, EXIT_INCOMPLETE, EXIT_PASS
from research1.adapters import check_harbor_import, proxy_url, with_llm_provider_env
from research1.config import load_json, load_pilot, resolve_path, train_task_names
from research1.versions import verify_against_snapshot


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    blocking: bool = True


@dataclass
class PreflightReport:
    status: str
    checks: list[CheckResult] = field(default_factory=list)
    config_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "config_hash": self.config_hash,
            "checks": [
                {
                    "name": c.name,
                    "ok": c.ok,
                    "detail": c.detail,
                    "blocking": c.blocking,
                }
                for c in self.checks
            ],
        }

    @property
    def exit_code(self) -> int:
        if self.status == "PASS":
            return EXIT_PASS
        if self.status == "FAIL":
            return EXIT_FAIL
        return EXIT_INCOMPLETE


def _proxy_reachable(url: str) -> tuple[bool, str]:
    parsed = urlparse(url)
    host = parsed.hostname or "172.17.0.1"
    port = parsed.port or 7890
    try:
        with socket.create_connection((host, port), timeout=3):
            return True, f"tcp ok {host}:{port}"
    except OSError as exc:
        return False, f"proxy unreachable {host}:{port}: {exc}"


def _disk_ok(path: Path, stop_gb: float) -> tuple[bool, str]:
    script = Path(__file__).resolve().parents[2] / "scripts/check_disk.py"
    proc = subprocess.run(
        ["python3", str(script), "--path", str(path), "--stop-gb", str(stop_gb)],
        capture_output=True,
        text=True,
    )
    detail = (proc.stdout or proc.stderr or "").strip()
    return proc.returncode == 0, detail


def _docker_ok() -> tuple[bool, str]:
    proc = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "docker info failed")[:300]
    return True, "docker info ok"


def run_preflight(config_path: Path | None = None) -> PreflightReport:
    cfg = load_pilot(config_path)
    report = PreflightReport(status="PASS", config_hash=cfg.get("_config_hash", ""))
    checks = report.checks

    # Disk
    stop_gb = float(cfg.get("disk_stop_gb_free", 15))
    ok, detail = _disk_ok(Path("/home/jinxu"), stop_gb)
    checks.append(CheckResult("disk", ok, detail))

    # Proxy
    px = cfg.get("proxy_url") or proxy_url()
    ok, detail = _proxy_reachable(px)
    checks.append(CheckResult("proxy", ok, f"{px} :: {detail}"))

    # Optional: HTTP via proxy to a public URL (non-blocking if TCP ok)
    try:
        req = urllib.request.Request("https://api.deepseek.com", method="HEAD")
        # just connectivity probe — may 4xx without key; that is fine
        urllib.request.urlopen(req, timeout=5)  # noqa: S310
        checks.append(CheckResult("egress", True, "api.deepseek.com reachable", blocking=False))
    except Exception as exc:  # noqa: BLE001
        checks.append(CheckResult("egress", False, f"egress probe: {exc}", blocking=False))

    # Provider
    try:
        prov = with_llm_provider_env(cfg.get("provider_mode"))
        keys_ok = bool(prov.api_key) and bool(prov.api_base) and bool(prov.model)
        detail = (
            f"provider={prov.provider_id} mode={prov.mode} model={prov.model} "
            f"base={prov.api_base} key_present={bool(prov.api_key)}"
        )
        checks.append(CheckResult("provider", keys_ok, detail))
    except Exception as exc:  # noqa: BLE001
        checks.append(CheckResult("provider", False, str(exc)))

    # Docker
    ok, detail = _docker_ok()
    checks.append(CheckResult("docker", ok, detail))

    # ModularRSI + harbor
    mr = resolve_path(cfg, "modularrsi_root")
    if not mr.is_dir():
        checks.append(CheckResult("modularrsi_root", False, f"missing {mr}"))
    else:
        checks.append(CheckResult("modularrsi_root", True, str(mr)))
        try:
            harbor_path = check_harbor_import(mr)
            checks.append(CheckResult("harbor_import", True, str(harbor_path)))
        except Exception as exc:  # noqa: BLE001
            checks.append(CheckResult("harbor_import", False, str(exc)))

    # Dataset
    ds = resolve_path(cfg, "dataset_tb")
    if not (ds / "tasks").is_dir():
        checks.append(
            CheckResult(
                "dataset",
                False,
                f"missing tasks under {ds} (expected ModularRSI_2000_Instances/tb)",
            )
        )
    else:
        try:
            trains = train_task_names(ds)
            n_tasks = sum(1 for _ in (ds / "tasks").iterdir() if _.is_dir())
            fix_git = (ds / "tasks" / "fix-git").is_dir()
            detail = (
                f"tasks={n_tasks} train={len(trains)} fix-git_in_train="
                f"{'fix-git' in trains} fix-git_present={fix_git}"
            )
            checks.append(CheckResult("dataset", len(trains) >= 8, detail))
        except Exception as exc:  # noqa: BLE001
            checks.append(CheckResult("dataset", False, str(exc)))

    # H0 hashes
    snap_path = resolve_path(cfg, "h0_snapshot")
    modules = resolve_path(cfg, "modules_root")
    if not snap_path.is_file():
        checks.append(CheckResult("h0_snapshot", False, f"missing {snap_path}"))
    elif not modules.is_dir():
        checks.append(CheckResult("h0_modules", False, f"missing {modules}"))
    else:
        snap = load_json(snap_path)
        ok, mismatches = verify_against_snapshot(modules, snap)
        detail = "hashes match snapshot" if ok else f"mismatches={mismatches[:5]}"
        checks.append(CheckResult("h0_hashes", ok, detail))
        commit_file = mr / ".upstream_commit"
        expected = snap.get("upstream_commit", "")
        if commit_file.is_file():
            got = commit_file.read_text().strip()
            checks.append(
                CheckResult(
                    "h0_commit",
                    got == expected,
                    f"got={got} expected={expected}",
                )
            )

    # h0.lock presence (written by P0)
    lock = resolve_path(cfg, "h0_lock") if cfg.get("h0_lock") else (
        Path(cfg["_config_path"]).parent / "h0.lock.json"
    )
    if lock.is_file():
        checks.append(CheckResult("h0_lock", True, str(lock), blocking=False))
    else:
        checks.append(CheckResult("h0_lock", False, f"missing {lock}", blocking=False))

    blocking_fail = [c for c in checks if c.blocking and not c.ok]
    if not blocking_fail:
        report.status = "PASS"
    else:
        # Missing dataset / provider / proxy → INCOMPLETE (external blockers)
        incomplete_names = {"dataset", "provider", "proxy", "disk", "harbor_import"}
        if all(c.name in incomplete_names for c in blocking_fail):
            report.status = "INCOMPLETE"
        elif any(c.name in {"h0_hashes", "h0_commit"} for c in blocking_fail):
            report.status = "FAIL"
        else:
            report.status = "INCOMPLETE"
    return report


def write_h0_lock(cfg: dict[str, Any] | None = None) -> Path:
    """Materialize configs/research1/h0.lock.json after verifying snapshot."""
    cfg = cfg or load_pilot()
    snap = load_json(resolve_path(cfg, "h0_snapshot"))
    modules = resolve_path(cfg, "modules_root")
    ok, mismatches = verify_against_snapshot(modules, snap)
    if not ok:
        raise RuntimeError(f"H0 hash verify failed: {mismatches[:10]}")
    prov = None
    try:
        prov = with_llm_provider_env(cfg.get("provider_mode"))
    except Exception:  # noqa: BLE001
        pass
    lock = {
        "lock_id": "h0.lock.v1",
        "protocol_id": "research1-v1",
        "h0_id": snap.get("h0_id"),
        "upstream_commit": snap.get("upstream_commit"),
        "modules_root": str(modules),
        "composer": snap.get("composer_for_h0", "static"),
        "baseline_impl_files": snap.get("baseline_impl_files"),
        "file_sha256": snap.get("file_sha256"),
        "proxy_url": cfg.get("proxy_url") or proxy_url(),
        "provider_mode": cfg.get("provider_mode", "auto"),
        "provider_at_lock": None
        if prov is None
        else {
            "provider_id": prov.provider_id,
            "api_base": prov.api_base,
            "model": prov.model,
        },
        "verified": True,
    }
    out = Path(cfg.get("h0_lock") or (Path(cfg["_config_path"]).parent / "h0.lock.json"))
    if not out.is_absolute():
        from research1.config import REPO_ROOT

        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    return out
