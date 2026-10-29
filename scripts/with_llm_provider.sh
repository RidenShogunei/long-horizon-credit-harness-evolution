#!/usr/bin/env bash
# Usage:
#   source scripts/with_llm_provider.sh           # export active provider into current shell
#   source scripts/with_llm_provider.sh tongji_agi
#   bash scripts/with_llm_provider.sh auto -- python3 -c 'import os; print(os.environ["API_BASE"])'
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
if [[ "${1:-}" == "--" ]]; then
  MODE=""
  shift
elif [[ $# -gt 0 && "${1:-}" != "--" ]]; then
  # optional mode/provider as first arg
  case "$1" in
    auto|tongji_agi|deepseek_official)
      MODE="$1"
      shift
      if [[ "${1:-}" == "--" ]]; then shift; fi
      ;;
  esac
fi

EXPORT_ARGS=(export-env --with-meta)
if [[ -n "$MODE" ]]; then
  EXPORT_ARGS+=(--mode "$MODE")
fi

# shellcheck disable=SC1091
eval "$(python3 "$REPO/scripts/llm_provider.py" "${EXPORT_ARGS[@]}")"

if [[ $# -gt 0 ]]; then
  exec "$@"
fi
