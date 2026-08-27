# ACT C Fixed-Point Agent — Rabo Case Deployment

The GitHub `agent_system` repository is the only deployment source. Training repositories, Episodes, rosbags, checkpoints, and training outputs are not part of deployment.

## Normal platform operation

Rabo starts the Case controller without user-supplied arguments. The repository `main.py` selects `act_c_policy`, imports `agents.act_c_policy`, and calls its exported `run()` lifecycle entrypoint. There is no separate Agent registry, decorator, or manifest in this project.

The user workflow is:

1. Open the Rabo Case backed by this repository revision.
2. Start the Case/Agent in the Rabo platform.
3. Observe `Agent loaded`, readiness gates, and then `DONE` or `FAULT` in platform logs.

No terminal, Python command, CLI flag, or separately started tool process is part of normal operation.

At startup the Agent loads the repository-owned ONNX model, discovers the TOP camera, initializes the same arm/hand bundles used by the successful C Expert, validates live state26 and one ONNX inference, and only then constructs the serial arm executor and begins the task.

## New Rabo / Case synchronization

Connect or update the Case from the same GitHub repository and revision, then let the Rabo platform install/build the Case dependencies from the root `requirements.txt`. Select/start the Case normally. `onnxruntime` is declared in that platform dependency file.

The camera defaults to `"auto"`. It selects the unique RGB `sensor_msgs/msg/Image` topic whose entity is not one of the discovered seven-joint arm entities. Zero or multiple legal candidates produce a fail-closed `FAULT`; a platform developer may then set `camera_topic` explicitly in `config/act_c_policy.json` and commit that Case configuration.

Arm and hand construction is not name-discovered and does not use guessed aliases. It directly reuses `make_right_bundle()` and `make_left_bundle()` from the successfully executed C Expert stack, including its verified `DEVICE_IDS` source and shutdown helpers.

## Development-only diagnostics

`tools/run_act_c_policy_rabo.py` and `tools/check_act_c_environment.py` remain available to developers for diagnostics. They are not formal deployment entrypoints and are not required for normal Rabo Agent startup.

## Repository-owned deployment assets

- `agents/act_c_policy/`
- `models/act_c_fixed_point_v1.onnx`
- `models/act_c_fixed_point_v1.json`
- `config/act_c_policy.json`
- `requirements.txt`
- `requirements-act-c.txt` (development convenience only)
- `tools/check_act_c_environment.py`
- `tools/run_act_c_policy_rabo.py`

Runtime JSONL files are written below repo-relative `logs/` and are not deployment assets.
