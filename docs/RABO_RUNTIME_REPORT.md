# Rabo Runtime Report

Generated: `2026-08-13T18:17:54+0800`

## Environment

- Hostname: `liyi`
- PWD: `/home/liyi/agent_system`
- Python: `3.10.12 (main, Jun 22 2026, 18:55:27) [GCC 11.4.0]`
- ROS_DISTRO: `humble`
- ROS_DOMAIN_ID: `None`
- RMW_IMPLEMENTATION: `None`

## Summary

- Cameras discovered: `0`
- Robot state topics: `0`
- Robot control interfaces: `0`
- Can read nut ground-truth pose: `NO`
- Can control arm: `NOT FOUND`
- Can control hand: `NOT FOUND`

## Cameras

- No RGB camera frames discovered.

## Robot State

- No sampled robot state topic.

## Robot Control Interfaces (Read-Only Discovery)

- No command/control interfaces found by ROS discovery.

No command/control/reset interface was called by this probe.

## ROS Discovery Raw Result

- ROS available through rclpy: `True`
- ROS nodes: `['rabo_runtime_readonly_probe']`
- Topic count: `2`
  - `/parameter_events` type=`['rcl_interfaces/msg/ParameterEvent']` publishers=`1` subscribers=`0`
  - `/rosout` type=`['rcl_interfaces/msg/Log']` publishers=`1` subscribers=`0`
- Service count: `6`
  - `/rabo_runtime_readonly_probe/describe_parameters` type=`['rcl_interfaces/srv/DescribeParameters']`
  - `/rabo_runtime_readonly_probe/get_parameter_types` type=`['rcl_interfaces/srv/GetParameterTypes']`
  - `/rabo_runtime_readonly_probe/get_parameters` type=`['rcl_interfaces/srv/GetParameters']`
  - `/rabo_runtime_readonly_probe/list_parameters` type=`['rcl_interfaces/srv/ListParameters']`
  - `/rabo_runtime_readonly_probe/set_parameters` type=`['rcl_interfaces/srv/SetParameters']`
  - `/rabo_runtime_readonly_probe/set_parameters_atomically` type=`['rcl_interfaces/srv/SetParametersAtomically']`

## Known Object IDs From Source

- `NUT_A_ID`: `thing_3eb64241-4166-4c4e-9144-edf733c885f1`
- `NUT_B_ID`: `thing_e937f633-faed-4b2e-b609-a7b80825a64a`
- `NUT_C_ID`: `thing_4a38ffae-a85b-43c9-8b20-abb0cf6dcac7`

## Object / Target Runtime Discovery

- Object candidates: `0`
- Target candidates: `0`
- Runtime pose availability is recorded in `outputs/runtime_probe/runtime_probe.json`.

## Reset / Episode Interfaces

- No reset interface found by ROS discovery.
- Static reset-related source matches exist; inspect JSON for exact file/line.

## Static Source Findings

- Total keyword matches: `312`
- Control/example matches: `25`
- SDK imports: `{"rabo_robocap": {"available": false, "error": "ModuleNotFoundError(\"No module named 'rabo_robocap'\")"}, "rabo_dev_kit": {"available": false, "error": "ModuleNotFoundError(\"No module named 'rabo_dev_kit'\")"}}`

## Raw JSON

Machine-readable report: `outputs/runtime_probe/runtime_probe.json`
