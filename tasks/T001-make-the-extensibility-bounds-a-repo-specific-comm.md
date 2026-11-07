---
id: T001
title: Make the extensibility bounds a repo-specific committed file
state: Draft
priority: low
type: refactor
risk: null
created: 2026-11-07
updated: 2026-11-07
depends_on: []
strategy: null
branch: null
worktree: null
---

# T001: Make the extensibility bounds a repo-specific committed file

State and metadata live in the frontmatter above and nowhere else.

## 1. Description
The extensibility-bounds file for hybrid-motion-planner and lerobot is one file outside any repo, `WLIURA-2026S/.claude/extensibility-bounds.md` (named by the plug-and-play rule and by hybrid-motion-planner's CLAUDE.md). Turn it into a repo-specific, committed `extensibility-bounds.md` in **lerobot**, dividing the existing axes so each repo keeps the axes whose seam lives in it. The workspace file stays outside the repos until both tasks (hybrid-motion-planner and lerobot) are done.

## 2. Ideation, reuse, extensions
- **Ideation / unclear points:** initial split proposal (to verify against the code): lerobot keeps the axes whose seam is in its code, most likely the environment-side half of "Gym controller vs policy action layout" (`EnvConfig`), but I did not find `align_controller_with_policy` on the checked-out branches of either repo, so the home of that axis needs checking first; the other axes (action representation, MP paths, pose metric, spline timing, hybrid controllers, MP rewrite) are hybrid-motion-planner code and go to that repo. This repo is a fork (`lerobot-act-segment`) with upstream-style CLAUDE.md and AGENTS.md; check how a bounds file should sit beside them.
  Open: file location (repo root or `.claude/`); how agents find bounds that span both repos (a pointer from each file to the other); update the plug-and-play rule and each repo's CLAUDE.md to name the new files; what becomes of the workspace file (an index, or removed). Do the matching task in the other repo in the same pass.
- **Reuse check:** <!-- existing code to reuse, with paths -->
- **Future extensions:** <!-- what must stay interchangeable, what is out of scope; bounds file updated? -->

## 3. Contract
- **Intent:**
- **Acceptance criteria:**
- **Constraints:**
- **Open decisions:**

## 4. Key components
```mermaid
flowchart LR
  A --> B
```

| Assumption | Source (stated by user / agent's choice) |
|---|---|
| | |

## 5. Risk tier
<!-- proposed: low | med | high, with a one-line reason. Confirmed value goes in registry.yaml. -->

## 6. Test plan
<!-- Hierarchical: behaviour → concrete normal and edge cases → expected outcome.
     Risky refactor: characterization tests and a keep / bug / unclear list first. -->

## 7. Todos
- [ ] 1. Draft the contract, including the section-by-section split, for approval.

## 8. Decision log
| Date | Decision | Category | Chosen | Why | Strongest alternative | Cost to change later | Status |
|---|---|---|---|---|---|---|---|
| | | pre-approved / reversible / material | | | | | open / decided |

## 9. Approvals
| Date | What was approved | Note |
|---|---|---|
| | | |

## 10. Review packet
<!-- Written when verifying, after the size and duplication check: coverage, tests, change map, decisions, risks, runtime evidence, demonstration of new features (link to tasks/<ID>-demo/, code commit and recreate commands, input files, script and output verbatim, reading and limits), size and refactor. -->

## 11. Revision notes
<!-- One entry per failing check, finding or feedback item: what, and what changed in the contract/tests/code. -->
