# .claude/ — Claude Code configuration

Team-shared Claude Code setup for this NeurIPS 2026 / EMNLP 2026 research repo. Everything here is committed; per-machine overrides go in `settings.local.json` (gitignored).

## Workflow this is built around

```
[ LOCAL (this machine) ]           [ KAGGLE H100 ]
 Claude edits expNN_*.py  ───►  user uploads/clones to Kaggle
                                    │
                                    ▼
                              notebook runs it, produces log
                                    │
 user pastes log back  ◄───────────┘
        │
        ▼
 Claude runs /analyze-log
        │
        ▼
 /update-tracker appends a row to the right tracker.md
```

**Local = code, scaffold, plan, analyze pasted logs. Kaggle = run.** Claude never runs `python Exp_*/expNN.py`, `pytest`, or any training loop locally. The deny list in `settings.json` enforces this.

## Layout

```
.claude/
├── settings.json             # shared permissions, env, hooks (committed)
├── settings.local.json       # per-machine overrides (gitignored)
├── commands/                 # slash commands — invoke with /<name>
│   ├── new-experiment.md     # /new-experiment <suite> <expNN> "<desc>"
│   ├── kaggle-cell.md        # /kaggle-cell <path.py> [--smoke]
│   ├── analyze-log.md        # /analyze-log [path or paste inline]
│   ├── update-tracker.md     # /update-tracker <expNN> <bench> <source>
│   ├── compare-paper.md      # /compare-paper <bench> <task> <value>
│   ├── tracker-status.md     # /tracker-status
│   ├── neurips-check.md      # /neurips-check <paper.tex>
│   └── hf-dataset-peek.md    # /hf-dataset-peek <bench> [split] [N]
├── agents/                   # specialist subagents — Claude delegates to these
│   ├── results-analyzer.md       # pasted log → summary row + verdict
│   ├── paper-baseline-auditor.md # catch cited-number drift
│   ├── experiment-planner.md     # idea → concrete expNN plan
│   └── neurips-writer.md         # draft/revise paper sections, grounded in trackers
└── hooks/
    └── guard_destructive.py  # blocks rm -rf / force-push / reset --hard on main
```

## Typical session

1. `/tracker-status` — "where are we right now?"
2. Ask the `experiment-planner` agent for the next idea; approve it
3. `/new-experiment Exp_DM exp31_myidea "one-liner"` — scaffold the file
4. Edit the scaffold locally (Claude helps)
5. `/kaggle-cell Exp_DM/exp31_myidea.py` — get the Kaggle cells to paste
6. Run on Kaggle; copy the log back
7. `/analyze-log` (paste the log) → summary + tracker row preview
8. `/update-tracker exp31 aicd <pasted metrics>` — append row
9. `/compare-paper aicd T1 <value>` — Δ-vs-paper verdict

## Permissions philosophy

- **Allow-list** everything read-only and lint-level: `ruff`, `black`, `git status/diff/log`, `gh pr`, `Read/Glob/Grep`, HF/arxiv WebFetch.
- **Deny-list** the two things that would break the workflow:
  - Running an experiment script locally (`python Exp_DM/*` etc.) — this belongs on Kaggle, not the laptop.
  - Destructive ops on research artifacts (force-push, `rm -rf`, hard reset to main, edits to NeurIPS `.sty`).
- Everything else prompts. Prefer adding to `settings.local.json` first to try it out, then promote to `settings.json` if shareable.

## Hooks

- **UserPromptSubmit**: prepends a one-line reminder each turn: `LOCAL=code+analyze, KAGGLE=run` + metric mapping.
- **PreToolUse (Bash)**: `.claude/hooks/guard_destructive.py` blocks `rm -rf`, force-push, hard reset to `origin/main`, and `rm` against `results/` `logs/` `codet_m4_checkpoints/` `Exp_*/` `docs/references/`.

## Extending

- **Add a slash command**: drop `commands/<name>.md` with YAML frontmatter (`description`, `argument-hint`, `allowed-tools`). Body is the prompt template; `$ARGUMENTS` interpolates the user's args.
- **Add an agent**: drop `agents/<name>.md` with frontmatter (`name`, `description`, `tools`, `model`). Write `description` so Claude's router picks it up — include *when to use it* in plain English.
- **Add a permission**: try it in `settings.local.json` first; promote to `settings.json` once it's confirmed safe and team-useful.

## Design notes

- **All tracker-writing is gated** through `/update-tracker` so no agent silently rewrites history. Historical rows are immutable by project convention.
- **`results-analyzer`'s primary input is pasted text**, not a file or process — matches the real workflow.
- **`/kaggle-cell` emits text only** — it never runs `python` or `!pip` itself, just produces the cells for the user to paste.
