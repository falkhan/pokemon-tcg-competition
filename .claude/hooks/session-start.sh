#!/bin/bash
set -euo pipefail

# Only needed in Claude Code on the web containers; local setups manage their own env.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

export PATH="$HOME/.local/bin:$PATH"

cd "$CLAUDE_PROJECT_DIR"

# Installs the pinned Python (>=3.12) and all locked dependencies, dev group
# included (pytest). Idempotent: no-ops when uv.lock is already satisfied.
uv sync
