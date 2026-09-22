#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

HOST="${UI_HOST:-127.0.0.1}"
PORT="${UI_PORT:-7860}"
OPEN_BROWSER="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:?missing value for --host}"
      shift 2
      ;;
    --port)
      PORT="${2:?missing value for --port}"
      shift 2
      ;;
    --no-open)
      OPEN_BROWSER="0"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--host 127.0.0.1] [--port 7860] [--no-open]"
      exit 1
      ;;
  esac
done

if [[ -d ".venv_agent312" ]]; then
  # shellcheck disable=SC1091
  source ".venv_agent312/bin/activate"
elif [[ -d ".venv_agent" ]]; then
  # shellcheck disable=SC1091
  source ".venv_agent/bin/activate"
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

export DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
export DATA_CSV_PATH="${DATA_CSV_PATH:-${ROOT_DIR}/data/GLC25_PA_metadata_train.csv}"
export TIFF_ROOT="${TIFF_ROOT:-${ROOT_DIR}/data/SatelitePatches/PA-train}"

PROVIDER="${AGENT_LLM_PROVIDER:-ollama}"
MODEL="${OPENROUTER_MODEL:-${OLLAMA_MODEL:-qwen3:8b}}"
if [[ "${PROVIDER}" == "openrouter" ]]; then
  MODEL="${OPENROUTER_MODEL:-google/gemma-4-31b-it:free}"
fi

if [[ "${PROVIDER}" == "openrouter" ]]; then
  if [[ -z "${OPENROUTER_API_KEY:-}" || "${OPENROUTER_API_KEY:-}" == *"..."* ]]; then
    echo "[warn] OPENROUTER_API_KEY missing or placeholder; OpenRouter calls will fail."
  fi
fi

URL="http://${HOST}:${PORT}"
echo "[info] root=${ROOT_DIR}"
echo "[info] provider=${PROVIDER} model=${MODEL}"
echo "[info] data_csv=${DATA_CSV_PATH}"
echo "[info] launching UI at ${URL}"

if [[ "${OPEN_BROWSER}" == "1" ]] && command -v open >/dev/null 2>&1; then
  (
    sleep 2
    open "${URL}" >/dev/null 2>&1 || true
  ) &
fi

PYTHON_CMD="${MCP_SERVER_PYTHON:-python}"
exec "${PYTHON_CMD}" MCP_server/mcp_web_ui.py --host "${HOST}" --port "${PORT}"
