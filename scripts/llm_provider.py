#!/usr/bin/env python3
"""CLI for LLM provider switching / weekly quota ledger.

Examples:
  python3 scripts/llm_provider.py status
  python3 scripts/llm_provider.py use auto
  python3 scripts/llm_provider.py use tongji_agi
  python3 scripts/llm_provider.py use deepseek_official
  python3 scripts/llm_provider.py export-env          # shell-safe KEY=value lines
  python3 scripts/llm_provider.py set-weekly-limit tongji_agi 2000000
  python3 scripts/llm_provider.py record tongji_agi --prompt 100 --completion 50
  python3 scripts/llm_provider.py exhaust tongji_agi --reason 'manual: weekly cap hit'
  python3 scripts/llm_provider.py clear-exhausted tongji_agi
  python3 scripts/llm_provider.py probe               # tiny live call via active provider
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from llm_provider_lib import (  # noqa: E402
    clear_exhausted,
    load_provider_config,
    looks_like_quota_error,
    mark_exhausted,
    record_usage,
    resolve_provider,
    set_mode,
    status_dict,
)


def cmd_status(_: argparse.Namespace) -> int:
    print(json.dumps(status_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_use(args: argparse.Namespace) -> int:
    set_mode(args.provider)
    print(json.dumps(status_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_export_env(args: argparse.Namespace) -> int:
    r = resolve_provider(mode=args.mode)
    exports = r.harbor_exports()
    # Optional: also export a non-secret summary line for logs.
    if args.with_meta:
        print(f"# provider={r.provider_id} mode={r.mode} model={r.harbor_model} base={r.api_base}")
    for k, v in exports.items():
        print(f"export {k}={shlex.quote(v)}")
    return 0


def cmd_set_weekly_limit(args: argparse.Namespace) -> int:
    """Persist weekly limit into hermes .env (not the repo)."""
    cfg = load_provider_config()
    if args.provider not in cfg["providers"]:
        raise SystemExit(f"unknown provider {args.provider}")
    env_key = cfg["providers"][args.provider].get("weekly_token_limit_env")
    if not env_key:
        raise SystemExit(f"provider {args.provider} has no weekly_token_limit_env")
    secrets = Path(cfg.get("secrets_file") or "/home/jinxu/.hermes/.env")
    lines = secrets.read_text().splitlines() if secrets.is_file() else []
    out = []
    seen = False
    for line in lines:
        if line.startswith(env_key + "="):
            out.append(f"{env_key}={int(args.tokens)}")
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append(f"{env_key}={int(args.tokens)}")
    secrets.write_text("\n".join(out).rstrip() + "\n")
    secrets.chmod(0o600)
    print(json.dumps({"updated_env": env_key, "weekly_token_limit": int(args.tokens)}, indent=2))
    print(json.dumps(status_dict()["providers"][args.provider], indent=2))
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    state = record_usage(
        args.provider,
        prompt_tokens=args.prompt,
        completion_tokens=args.completion,
        total_tokens=args.total,
        meta={"source": "cli"},
    )
    print(json.dumps({"ok": True, "week_entry": state["weeks"]}, indent=2)[:2000])
    return 0


def cmd_exhaust(args: argparse.Namespace) -> int:
    mark_exhausted(args.provider, reason=args.reason or "manual exhaust")
    print(json.dumps(status_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    clear_exhausted(args.provider)
    print(json.dumps(status_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    r = resolve_provider(mode=args.mode)
    body = json.dumps(
        {
            "model": r.api_model_id,
            "messages": [{"role": "user", "content": "只回复：ok"}],
            "max_tokens": 16,
            "temperature": 0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        r.api_base.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {r.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        cfg = load_provider_config()
        if looks_like_quota_error(raw, cfg["providers"][r.provider_id]):
            mark_exhausted(r.provider_id, reason=raw)
            print(json.dumps({"ok": False, "exhausted": True, "provider": r.provider_id, "error": raw[:400]}, indent=2))
            return 3
        print(json.dumps({"ok": False, "provider": r.provider_id, "error": raw[:400]}, indent=2))
        return 2

    usage = data.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    record_usage(r.provider_id, prompt_tokens=prompt, completion_tokens=completion, meta={"source": "probe"})
    content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
    print(
        json.dumps(
            {
                "ok": True,
                "provider": r.provider_id,
                "api_base": r.api_base,
                "model": data.get("model"),
                "content": content,
                "usage": usage,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("status")
    p.set_defaults(func=cmd_status)

    p = sp.add_parser("use")
    p.add_argument("provider", choices=["auto", "tongji_agi", "deepseek_official"])
    p.set_defaults(func=cmd_use)

    p = sp.add_parser("export-env")
    p.add_argument("--mode", default=None)
    p.add_argument("--with-meta", action="store_true")
    p.set_defaults(func=cmd_export_env)

    p = sp.add_parser("set-weekly-limit")
    p.add_argument("provider", choices=["tongji_agi", "deepseek_official"])
    p.add_argument("tokens", type=int)
    p.set_defaults(func=cmd_set_weekly_limit)

    p = sp.add_parser("record")
    p.add_argument("provider")
    p.add_argument("--prompt", type=int, default=0)
    p.add_argument("--completion", type=int, default=0)
    p.add_argument("--total", type=int, default=None)
    p.set_defaults(func=cmd_record)

    p = sp.add_parser("exhaust")
    p.add_argument("provider")
    p.add_argument("--reason", default="manual")
    p.set_defaults(func=cmd_exhaust)

    p = sp.add_parser("clear-exhausted")
    p.add_argument("provider", nargs="?", default=None)
    p.set_defaults(func=cmd_clear)

    p = sp.add_parser("probe")
    p.add_argument("--mode", default=None)
    p.set_defaults(func=cmd_probe)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
