#!/usr/bin/env python3
"""PreToolUse guard for Bash.

Blocks destructive commands against result/checkpoint/log directories
that would wipe experiment artifacts. Runs on every Bash tool call.

Input: JSON on stdin with shape {"tool_input": {"command": "..."}}.
Exit 0 => allow. Exit 2 => block with stderr as explanation.
"""
from __future__ import annotations

import json
import re
import sys

DANGER_PATTERNS = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f?|-[a-zA-Z]*f[a-zA-Z]*r)"),
     "rm -rf blocked — confirm with user first"),
    (re.compile(r"\bgit\s+push\s+.*--force\b"),
     "force-push blocked — confirm with user first"),
    (re.compile(r"\bgit\s+push\s+.*\s-f\b"),
     "force-push blocked — confirm with user first"),
    (re.compile(r"\bgit\s+reset\s+--hard\b.*\borigin/main\b"),
     "hard reset to origin/main blocked — confirm with user first"),
    (re.compile(r"\bgit\s+clean\s+-[a-zA-Z]*f"),
     "git clean -f blocked — would delete untracked experiment artifacts"),
    (re.compile(r"\b(results|logs|codet_m4_checkpoints)/\S*\s*$", re.MULTILINE),
     "direct deletion under results/logs/checkpoints blocked"),
]

PROTECTED_PATH_DELETE = re.compile(
    r"\brm\s+[^|;&]*\b(results|logs|codet_m4_checkpoints|Exp_DM|Exp_CodeDet|Exp_Climb|Exp_TK|docs/references)\b"
)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    cmd = (payload.get("tool_input") or {}).get("command", "")
    if not cmd:
        return 0

    for pat, msg in DANGER_PATTERNS:
        if pat.search(cmd):
            sys.stderr.write(f"[guard] {msg}\n  command: {cmd}\n")
            return 2

    if PROTECTED_PATH_DELETE.search(cmd):
        sys.stderr.write(
            "[guard] deletion targets a protected research path "
            "(experiment dirs, results, logs, checkpoints, references).\n"
            f"  command: {cmd}\n"
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
