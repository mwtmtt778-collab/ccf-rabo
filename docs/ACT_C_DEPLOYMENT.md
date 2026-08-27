# ACT C Fixed-Point Agent — GitHub Deployment

This deployment uses only files committed to the `agent_system` repository. The training repository, Episodes, rosbags, checkpoints, and local training outputs are not required.

## Current Rabo

```bash
cd /workspace/agent_system
git pull origin main
python3 -m pip install -r requirements-act-c.txt
python3 tools/check_act_c_environment.py --discover
python3 -u tools/run_act_c_policy_rabo.py --dry-run
```

Only after the dry-run reports `PASS`, start the formal Agent:

```bash
python3 -u main.py act_c_policy
```

For the retained explicit MVP debug entrypoint:

```bash
python3 -u tools/run_act_c_policy_rabo.py --execute-mvp
```

The legacy `--execute` entrypoint remains disabled and fail-closed.

## New Rabo

```bash
git clone <agent_system-repo-url> agent_system
cd agent_system
python3 -m pip install -r requirements-act-c.txt
python3 tools/check_act_c_environment.py --discover
```

The default camera setting is `"auto"`. It selects the unique RGB `sensor_msgs/msg/Image` topic whose entity is not one of the discovered seven-joint arm entities. If discovery reports zero or multiple legal candidates, edit only:

```text
config/act_c_policy.json
```

Set `camera_topic` to one of the exact image topics printed by the checker. If the new scene does not use the standard Rabo logical device names, update the four arm/hand name fields in the same JSON; do not edit Python.

Then validate again:

```bash
python3 tools/check_act_c_environment.py --discover
python3 -u tools/run_act_c_policy_rabo.py --dry-run
```

After dry-run PASS:

```bash
python3 -u main.py act_c_policy
```

## Repository-owned deployment assets

- `agents/act_c_policy/`
- `models/act_c_fixed_point_v1.onnx`
- `models/act_c_fixed_point_v1.json`
- `config/act_c_policy.json`
- `requirements-act-c.txt`
- `tools/check_act_c_environment.py`
- `tools/run_act_c_policy_rabo.py`

Runtime JSONL files are written below repo-relative `logs/` and are not deployment assets.
