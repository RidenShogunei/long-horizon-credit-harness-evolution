#!/usr/bin/env bash
# Smoke H0 (ModularRSI static baseline modules) on one light task, twice.
# Does not modify /home/jinxu/meta-harness-repro.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UP="${MODULARRSI_ROOT:-/home/jinxu/arms_cache/modularrsi/ModularRSI}"
PYTHON_BIN="${PYTHON_BIN:-$UP/.venv/bin/python}"
MODULES_ROOT="${MODULES_ROOT:-$UP/generations/merged_active/gen_0/modules}"
TASK_NAME="${TASK_NAME:-fix-git}"
RUN_TAG="${RUN_TAG:-h0_smoke_$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$REPO/experiments/$RUN_TAG}"

python3 "$REPO/scripts/check_disk.py"

[[ -x "$PYTHON_BIN" ]] || { echo "missing ModularRSI venv python: $PYTHON_BIN" >&2; exit 2; }
[[ -d "$MODULES_ROOT" ]] || { echo "missing modules: $MODULES_ROOT" >&2; exit 2; }

# Load DeepSeek key without printing it.
# shellcheck disable=SC1091
eval "$(python3 - <<'PY'
from pathlib import Path
import shlex
wanted = {"DEEPSEEK_API_KEY"}
for line in Path("/home/jinxu/.hermes/.env").read_text().splitlines():
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    if k in wanted:
        print(f"export {k}={shlex.quote(v.strip().strip(chr(34)).strip(chr(39)))}")
PY
)"
[[ -n "${DEEPSEEK_API_KEY:-}" ]] || { echo "DEEPSEEK_API_KEY missing" >&2; exit 2; }

export HARBOR_EVO_API_KEY="$DEEPSEEK_API_KEY"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
export API_BASE="${API_BASE:-https://api.deepseek.com}"
export HARBOR_MODEL_INFO="${HARBOR_MODEL_INFO:-{\"max_input_tokens\":176000,\"max_output_tokens\":16000,\"input_cost_per_token\":0.0,\"output_cost_per_token\":0.0}}"
export PYTHONDONTWRITEBYTECODE=1

MODEL="${MODEL:-openai/deepseek-flash}"
COMPOSER_NAME="${COMPOSER_NAME:-static}"
SOLVER_TEMPERATURE="${SOLVER_TEMPERATURE:-0.7}"
ENVIRONMENT="${ENVIRONMENT:-docker}"
DATASET="${DATASET:-terminal-bench@2.0}"
MAX_TURNS="${MAX_TURNS:-200}"
ATTEMPTS="${ATTEMPTS:-2}"

# Verifier downloads (uv/pytest) need host proxy. Bridge gateway avoids
# depending on host.docker.internal inside ModularRSI's older Harbor.
DOCKER_GW="$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null || true)"
PROXY_URL="${PROXY_URL:-http://${DOCKER_GW:-172.17.0.1}:7890}"
NO_PROXY_LIST="${NO_PROXY_LIST:-localhost,127.0.0.1,::1,api.deepseek.com}"

mkdir -p "$OUT_ROOT"
cp "$REPO/configs/protocol.v0.json" "$OUT_ROOT/"
cp "$REPO/configs/roles.v0.json" "$OUT_ROOT/"
cp "$REPO/configs/h0.snapshot.v0.json" "$OUT_ROOT/"
cat >"$OUT_ROOT/runtime.env.json" <<EOF
{
  "protocol_id": "research1-v0",
  "model": "$MODEL",
  "composer_name": "$COMPOSER_NAME",
  "solver_temperature": $SOLVER_TEMPERATURE,
  "dataset": "$DATASET",
  "task_name": "$TASK_NAME",
  "modules_root": "$MODULES_ROOT",
  "proxy_url": "$PROXY_URL",
  "attempts": $ATTEMPTS
}
EOF

"$PYTHON_BIN" - <<PY
from pathlib import Path
import harbor
expected = (Path("$UP") / "src/harbor/__init__.py").resolve()
actual = Path(harbor.__file__).resolve()
assert actual == expected, (actual, expected)
print("harbor_ok", actual)
print("proxy", "$PROXY_URL")
PY

run_once() {
  local attempt="$1"
  local out="$OUT_ROOT/attempt_${attempt}"
  mkdir -p "$out"
  echo "=== attempt $attempt -> $out ==="
  # Call Harbor directly so we can pass verifier/agent proxy env.
  # Keep the same agent kwargs as scripts/evaluate.sh.
  "$PYTHON_BIN" -m harbor.cli.main run --yes \
    -d "$DATASET" \
    -a terminus-2-modular \
    -m "$MODEL" \
    -e "$ENVIRONMENT" \
    -n 1 \
    -k 1 \
    -o "$out" \
    -i "$TASK_NAME" \
    --ak "modules_root=$MODULES_ROOT" \
    --ak "api_base=$API_BASE" \
    --ak "max_turns=$MAX_TURNS" \
    --ak "model_info=$HARBOR_MODEL_INFO" \
    --ak "temperature=$SOLVER_TEMPERATURE" \
    --ak "locked_module_type=tools" \
    --ak "composer_scope=all" \
    --ak "composer_name=$COMPOSER_NAME" \
    --ve "HTTP_PROXY=$PROXY_URL" \
    --ve "HTTPS_PROXY=$PROXY_URL" \
    --ve "http_proxy=$PROXY_URL" \
    --ve "https_proxy=$PROXY_URL" \
    --ve "NO_PROXY=$NO_PROXY_LIST" \
    --ve "no_proxy=$NO_PROXY_LIST" \
    --ae "HTTP_PROXY=$PROXY_URL" \
    --ae "HTTPS_PROXY=$PROXY_URL" \
    --ae "http_proxy=$PROXY_URL" \
    --ae "https_proxy=$PROXY_URL" \
    --ae "NO_PROXY=$NO_PROXY_LIST" \
    --ae "no_proxy=$NO_PROXY_LIST"
}

for ((i=1; i<=ATTEMPTS; i++)); do
  run_once "$i"
done

echo "smoke outputs: $OUT_ROOT"
