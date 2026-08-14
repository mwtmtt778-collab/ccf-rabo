# Git Workflow Cleanup Report

## Current Commit

- Commit before cleanup: `340fe1d78d14d44eb6e642bfdc26dfbdef85fed5`
- Branch: `main`

## Git Remotes

- `origin`: `git@github-ccf-pc2:mwtmtt778-collab/ccf-rabo.git`

This local checkout uses `origin` for GitHub. The cloud checkout may use
`github` for the same repository and `origin` for the Rabo platform repository.

## Problems Found

- Runtime scripts wrote generated Markdown reports into tracked `docs/` files.
- `outputs/` contained generated JSON/PPM artifacts that were tracked by Git.
- `logs/` was partially ignored through `*.log`, but the directory itself was not ignored.
- Running `tools/run_three_nut_expert.py` could dirty `docs/RABO_THREE_NUT_EXPERT_IMPLEMENTATION_REPORT.md`.

## Changes Made

- Appended runtime artifact rules to `.gitignore`:
  - `logs/`
  - `outputs/`
  - Python runtime files
- Changed runtime report defaults:
  - Three-nut Expert report now writes to `outputs/three_nut_expert/latest_report.md`.
  - Camera test report now writes to `outputs/camera_test/latest_report.md`.
  - Runtime interface map report now writes to `outputs/runtime_interface_map/latest_report.md`.
  - Runtime probe report now writes to `outputs/runtime_probe/latest_report.md`.
- Kept `docs/RABO_THREE_NUT_EXPERT_IMPLEMENTATION_REPORT.md` as a fixed implementation document.
- Added `docs/VERSIONING.md`.
- Added `tools/git_checkpoint.sh`.
- Added `tools/create_milestone_tag.sh`.
- Removed generated `outputs/` files from Git tracking with `git rm --cached -r outputs`; files remain locally.

## Generated Runtime Paths

Expert report:
`outputs/three_nut_expert/latest_report.md`

Expert JSON:
`outputs/three_nut_expert/three_nut_expert_result.json`

Expert log:
`logs/three_nut_expert.log`

Camera report:
`outputs/camera_test/latest_report.md`

Runtime interface map report:
`outputs/runtime_interface_map/latest_report.md`

Runtime probe report:
`outputs/runtime_probe/latest_report.md`

## Git Dirty Test

Before dry-run:

```text
source changes from this cleanup were present; historical untracked manual files were already present.
```

Command:

```bash
python3 tools/run_three_nut_expert.py --mode single --nut B --trials 1 --no-jitter
```

After dry-run:

```text
git status --short was unchanged compared with the pre-run status.
```

Result:

PASS. The dry-run writes only ignored `outputs/` and `logs/` runtime artifacts and does not modify tracked `docs/` files.

## Version Strategy

### main

Current organized and reproducible mainline code.

### expert-dev

Expert development and debugging branch. Create later with:

```bash
git switch -c expert-dev
```

### milestone tags

Use milestone tags only after real validation. Dry-run success and SDK/API success are not enough for `*-pass` tags.

## Current Recommended Milestone

Current development stage:
single-nut physical grasp validation.

Current stable baseline:
Runtime / SDK interface mapping is available; single-nut B is not yet stable enough for a pass tag.

Next milestone:
`v0.3-single-nut-b-pass`

Only create it after single-nut B truly reaches stable physical grasp success.

Recommended tags to consider later:

- `v0.1-runtime-pass`
- `v0.2-camera-interface-pass`

Do not create:

- `v0.3-single-nut-b-pass`
- `v0.4-three-nut-expert-pass`

until the corresponding physical task validation passes.

## Modified Files

- `.gitignore`
- `docs/GIT_WORKFLOW_REPORT.md`
- `docs/RABO_THREE_NUT_EXPERT_IMPLEMENTATION_REPORT.md`
- `docs/VERSIONING.md`
- `tools/create_milestone_tag.sh`
- `tools/git_checkpoint.sh`
- `tools/inspect_rabo_runtime.py`
- `tools/map_rabo_runtime_interfaces.py`
- `tools/run_three_nut_expert.py`
- `tools/test_camera_stream.py`
- `outputs/*` removed from Git tracking
