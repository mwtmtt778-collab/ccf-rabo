# Project Versioning

## Branches

### main

Represents the current organized and reproducible mainline code.

Do not leave long-lived failed experiments or half-finished work on `main`.

### expert-dev

Used for Expert development and debugging.

If the branch does not exist yet, do not force a branch switch in the middle of
work. Create it later with:

```bash
git switch -c expert-dev
```

## Milestone Tags

Use tags for validated milestones.

Recommended tags:

- `v0.1-runtime-pass`: Rabo SDK / runtime state reading validated.
- `v0.2-camera-interface-pass`: Camera / Runtime Interface Mapping basically complete.
- `v0.3-single-nut-b-pass`: Single nut B physical grasp stable.
- `v0.4-three-nut-expert-pass`: Full three-nut Expert task stable.
- `v0.5-recorder-replay-pass`: Recorder + Replay validated.
- `v0.6-act-overfit-pass`: ACT overfits a small simulation episode set.
- `v1.0-simulation-baseline`: Full Rabo to Dataset to ACT to Rabo baseline.

## Tag Rule

Only create a milestone tag after the corresponding real test has passed.

Do not create a `*-pass` tag only because:

- code runs
- dry-run passes
- an API returns true

The corresponding physical or functional validation must pass first.
