@AGENTS.md

<!-- BEGIN FORK ADDITIONS: added after the fork from huggingface/lerobot (akirakudo901/lerobot-act-segment). Not part of upstream.
     Upstream's CLAUDE.md is a symlink to AGENTS.md; this file imports AGENTS.md unchanged (line above) and adds only the block below,
     so upstream changes to AGENTS.md merge without conflicts. -->
## Workflow (fork addition)
- Workflow definition: ../project-level-brainstorming/coding-workflow/coding-workflow.md
- Feature strategy (read when implementing a feature): ../project-level-brainstorming/coding-workflow/feature-strategy/default.md
- Tasks: `tasks/T*.md` (metadata in frontmatter) and `tasks/config.yaml`; list them with `python3 ../project-level-brainstorming/coding-workflow/scripts/tasks.py scan .`; new task: same script, `new`. Format in coding-workflow.md.
- Extensibility bounds: `../.claude/extensibility-bounds.md` for now (a repo-specific file is task T001).
- Repo-specific:
  - Overrides upstream's "prioritize `uv run`" for this workspace: run tests and scripts in the conda env `lerobot-act-libero` (see the workspace rule `.claude/rules/default-conda-env.md`; `-cpu` variant on the WSL machine).
  - This clone is imported by hybrid-motion-planner (`../lerobot/src`). Edit this workspace-root clone only, never the Sockeye mount copies. Changes made for an hmp task go on a branch `TYPE_OF_WORK/DESCRIPTION_TASKID` and should stay backward compatible.
  - Keep fork-specific edits to upstream files small, so syncing from upstream stays easy.

Agent instructions:
@../project-level-brainstorming/coding-workflow/agent-instructions/default.md
<!-- END FORK ADDITIONS -->
