#!/usr/bin/env bash
# Poetry-Tree ranking evaluation.
# Default input: data/poetry_tree/poetry_tree.json
# (download from https://huggingface.co/datasets/Zihao1/Poet-4B_training_data)
#   ./run_eval.sh gpt-4o
# Optional expert-critique JSON for ICL (not shipped):
#   export GOLD_JSON=/path/to/expert_critiques.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${1:-gpt-4o}"
OUT_DIR="${OUT_DIR:-output}"
mkdir -p "$OUT_DIR"

if [[ -z "${INPUT_JSON:-}" ]]; then
  INPUT_JSON="$REPO/data/poetry_tree/poetry_tree.json"
fi

ARGS=(
  --model "$MODEL"
  --rank-mode batch
  --workers "${WORKERS:-4}"
  --output "$OUT_DIR/poetry_ranking_${MODEL}.json"
)

if [[ -n "${GOLD_JSON:-}" && -f "${GOLD_JSON}" ]]; then
  ARGS+=(--gold "$GOLD_JSON" --use-gold)
  echo "Using expert-critique ICL: $GOLD_JSON"
else
  echo "No GOLD_JSON; ranking without expert few-shot demonstrations."
fi

if [[ ! -f "$INPUT_JSON" ]]; then
  echo "Missing tree JSON: $INPUT_JSON"
  exit 1
fi
ARGS+=(--input "$INPUT_JSON")

python "$ROOT/rank_tree.py" "${ARGS[@]}"
