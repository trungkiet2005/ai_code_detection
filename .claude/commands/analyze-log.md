---
description: Parse a pasted Kaggle training log and emit metric summary + tracker row
argument-hint: (paste the log after the command, or give a file path)
allowed-tools: Read, Grep, Glob, Task
---

# Analyze a Kaggle log

The user runs experiments on Kaggle and pastes the log back. This command delegates to the `results-analyzer` agent to produce a clean summary + tracker-ready row.

## Input

`$ARGUMENTS` is either:
- A file path (`logs/expNN_aicd_T1.log` or similar) — read it, then delegate
- **Or empty** — the user will paste the log text directly after the command. Use that inline content as the source.

## Steps

1. If `$ARGUMENTS` is a path and the file exists, read it first.
2. Delegate to the `results-analyzer` agent via the `Task` tool with:
   - The log text (pasted or file contents)
   - The expNN identifier (detect from the log header, the file name, or ask the user once if ambiguous)
   - The target benchmark + task (detect from log; confirm if ambiguous)
3. Surface the agent's output verbatim to the user.
4. Remind the user: running `/update-tracker` is a separate explicit step — this command does NOT write to any tracker file.

## Rules

- If the log is truncated (no final metric block), say so and ask whether the user wants partial-log analysis or to paste more.
- If the log contains a crash traceback, treat root-cause extraction as the primary output — not metric parsing.
- Never invent missing metrics. "missing" is a valid verdict for any field.
