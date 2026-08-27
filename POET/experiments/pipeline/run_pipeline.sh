#!/usr/bin/env bash
# Digital Poetry Party + expert-persona ranking (paper Table 1).
# Prompts: data/honglou/honglou_poems.json
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage: ./run_pipeline.sh [options]

  generate_poems.py       12-model poetry party on Honglou-Poem prompts
  rank_generated.py       expert-persona ranking (batch, four judges)

Options:
  --tasks <path>          generation prompts (default: data/honglou/honglou_poems.json)
  --poetry-out <path>     generation output
  --poetry-models <list>  comma-separated API model ids (sent as-is)
  --poetry-seed <int>
  --show-previous         let models see earlier poems (off by default)
  --hide-previous
  --rank-out <path>
  --rank-model <list>     comma-separated judges (paper: deepseek-reasoner,kimi-k2,gemini-2.5-pro,glm-4.6)
  --rank-mode <mode>      batch|separate (paper: batch)
  --gold <path>           optional expert-critique JSON for ICL (not in this repo)
  --skip-poetry
  --skip-rank
  --workers <int>
  -h, --help
EOF
}

TASKS="$REPO/data/honglou/honglou_poems.json"
POETRY_OUT="$ROOT/output/poetry_meet_results.json"
POETRY_MODELS=""
POETRY_SEED=""
SHOW_PREVIOUS=0
RANK_OUT="$ROOT/output/poetry_ranking.json"
RANK_MODEL="deepseek-reasoner,kimi-k2,gemini-2.5-pro,glm-4.6"
RANK_MODE="batch"
GOLD_PATH="${GOLD_JSON:-}"
POETRY_SKIP=0
RANK_SKIP=0
WORKERS=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks) TASKS="$2"; shift 2;;
    --poetry-out) POETRY_OUT="$2"; shift 2;;
    --poetry-models) POETRY_MODELS="$2"; shift 2;;
    --poetry-seed) POETRY_SEED="$2"; shift 2;;
    --show-previous) SHOW_PREVIOUS=1; shift;;
    --hide-previous) SHOW_PREVIOUS=0; shift;;
    --rank-out) RANK_OUT="$2"; shift 2;;
    --rank-model) RANK_MODEL="$2"; shift 2;;
    --rank-mode) RANK_MODE="$2"; shift 2;;
    --gold) GOLD_PATH="$2"; shift 2;;
    --skip-poetry) POETRY_SKIP=1; shift;;
    --skip-rank) RANK_SKIP=1; shift;;
    --workers) WORKERS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1"; usage; exit 1;;
  esac
done

echo ">>> repo: $REPO"
echo ">>> tasks: $TASKS"

if [[ "$POETRY_SKIP" -eq 0 ]]; then
  echo ">>> generate poems"
  CMD=(python "$ROOT/generate_poems.py" --tasks "$TASKS" --output "$POETRY_OUT" --workers "$WORKERS")
  if [[ -n "$POETRY_MODELS" ]]; then
    CMD+=(--models "$POETRY_MODELS")
  fi
  if [[ -n "$POETRY_SEED" ]]; then
    CMD+=(--seed "$POETRY_SEED")
  fi
  if [[ "$SHOW_PREVIOUS" -eq 1 ]]; then
    CMD+=(--show-previous)
  else
    CMD+=(--hide-previous)
  fi
  "${CMD[@]}"
else
  echo ">>> skip generation"
fi

if [[ -n "${ZSH_VERSION:-}" ]]; then
  IFS=',' read -A RANK_MODELS <<< "${RANK_MODEL}"
else
  IFS=',' read -ra RANK_MODELS <<< "${RANK_MODEL}"
fi
if [[ ${#RANK_MODELS[@]} -eq 0 ]]; then
  RANK_MODELS=("${RANK_MODEL}")
fi

for rank_model in "${RANK_MODELS[@]}"; do
  rank_model=$(echo "${rank_model:-}" | xargs)
  [[ -z "${rank_model:-}" ]] && continue

  RANK_OUT_WITH_MODEL="${RANK_OUT%.json}_${rank_model}.json"
  echo ">>> ranking model: ${rank_model}"

  if [[ "${RANK_SKIP:-0}" -eq 0 ]]; then
    RANK_CMD=(python "$ROOT/rank_generated.py"
      --poems "$POETRY_OUT"
      --output "$RANK_OUT_WITH_MODEL"
      --model "$rank_model"
      --rank-mode "$RANK_MODE"
      --workers "$WORKERS"
      --gold-samples 4)
    if [[ -n "$GOLD_PATH" && -f "$GOLD_PATH" ]]; then
      RANK_CMD+=(--gold "$GOLD_PATH")
      echo ">>> expert ICL: $GOLD_PATH"
    else
      echo ">>> no expert-critique JSON; ranking without ICL demonstrations"
    fi
    "${RANK_CMD[@]}"
  fi
done

echo ">>> done"
