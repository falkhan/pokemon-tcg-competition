#!/usr/bin/env zsh
# build_submission.sh — one-command submission pipeline (zsh twin of
# build_submission.ps1; all logic lives in Python via tcg/shipping.py).
#
# Build neural:      ./build_submission.sh [--checkpoint ppo_best.pt] [--deck lucario]
# Build rule agent:  ./build_submission.sh --agent rules
# Build + submit:    ./build_submission.sh --agent rules --message "M7.5 pilot fixes"
# (submitting is opt-in because Kaggle limits submissions per day)
set -e
setopt pipefail 2>/dev/null || set -o pipefail

COMPETITION="pokemon-tcg-ai-battle"
cd "$(dirname "$0")"

AGENT="neural"
MESSAGE=""
CHECKPOINT=""
DECK=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)      AGENT="$2"; shift 2 ;;
        --message)    MESSAGE="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --deck)       DECK="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
if [[ "$AGENT" != "neural" && "$AGENT" != "rules" ]]; then
    echo "--agent must be 'neural' or 'rules'" >&2; exit 2
fi

# Prefer the project venv (the ps1 uses .venv\Scripts\python.exe); fall back to uv.
if [[ -x .venv/bin/python ]]; then
    PY=(.venv/bin/python)
else
    PY=(uv run python)
fi

# Per-agent bundle layout (mirrors build_submission.ps1 — keep in sync).
if [[ "$AGENT" == "rules" ]]; then
    SUB_DIR="submission_rules"
    TAR_FILES=(main.py deck.csv cg rl)
    REQUIRED=(main.py deck.csv cg/api.py cg/libcg.so
              rl/__init__.py rl/combat.py rl/generic_pilot.py rl/turn_solver.py)
else
    # v2 neural bundle (M7.5): ships the actual rl/ encoder modules + the
    # feature matrix they fall back to (no polars/torch on Kaggle).
    SUB_DIR="submission"
    TAR_FILES=(main.py deck.csv policy_weights.npz card_features.npy cg rl)
    REQUIRED=(main.py deck.csv policy_weights.npz card_features.npy
              cg/api.py cg/utils.py cg/sim.py cg/libcg.so
              rl/__init__.py rl/combat.py rl/encoders.py rl/card_features.npy)
fi

EXPORT_ARGS=(--agent "$AGENT")
[[ -n "$CHECKPOINT" ]] && EXPORT_ARGS+=(--checkpoint "$CHECKPOINT")
[[ -n "$DECK" ]] && EXPORT_ARGS+=(--deck "$DECK")

echo "\e[36m[1/3] Export artifacts (tcg.shipping export ${EXPORT_ARGS[*]})\e[0m"
"${PY[@]}" -m tcg.shipping export "${EXPORT_ARGS[@]}"

echo "\e[36m[2/3] Gates (tcg.shipping gate --agent $AGENT)\e[0m"
"${PY[@]}" -m tcg.shipping gate --agent "$AGENT"

echo "\e[36m[3/3] Package\e[0m"
mkdir -p dist
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="dist/submission_${AGENT}_${STAMP}.tar.gz"
tar -czf "$OUT" -C "$SUB_DIR" "${TAR_FILES[@]}"

# The local gate can't catch a file missing from the BUNDLE (imports resolve from
# the project root here) -- so verify the archive contents explicitly.
CONTENTS=$(tar -tzf "$OUT")
for req in "${REQUIRED[@]}"; do
    if ! grep -qx "$req" <<< "$CONTENTS"; then
        echo "package is missing $req" >&2; exit 1
    fi
done

KB=$(( $(stat -c%s "$OUT") / 1024 ))
echo "\e[32m\nBUILD OK -> $OUT (${KB} KB)\e[0m"

if [[ -z "$MESSAGE" ]]; then
    echo "\e[33mNo --message given: skipping Kaggle upload. Submit later with:\e[0m"
    echo "\e[33m  kaggle competitions submit -c $COMPETITION -f $OUT -m '<message>'\e[0m"
    exit 0
fi

echo "\e[36m[4/4] Submitting to Kaggle\e[0m"
if [[ -x .venv/bin/kaggle ]]; then
    KAGGLE=(.venv/bin/kaggle)
else
    KAGGLE=(uv run kaggle)
fi
# No credential pre-check: the kaggle CLI finds credentials however they're
# configured (~/.kaggle, env vars, ...) and set -e aborts here if it fails.
"${KAGGLE[@]}" competitions submit -c "$COMPETITION" -f "$OUT" -m "$MESSAGE"

echo "\e[32m\nSUBMITTED. Recent submissions:\e[0m"
"${KAGGLE[@]}" competitions submissions -c "$COMPETITION" -v | head -4
