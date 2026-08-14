# Rabo Runtime Interface Map Report

## 1. Result

Overall: CHECK

This run was read-only: no motion APIs, no command publications, no action goals, no write services, no reset, and no scene/controller/camera parameter changes.

## 2. Environment

- Generated: 2026-08-14 14:35:04 +0800
- Hostname: cp-uf600987kmwepu0rhpfe-5dz4j
- Python: 3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]
- ROS_DISTRO: jazzy
- ROS_DOMAIN_ID: None
- RMW_IMPLEMENTATION: rmw_cyclonedds_cpp
- Runtime namespace: `/gs_1eebee6f37512bbc1d125b25511e912c`
- Output directory: `outputs/runtime_interface_map`
- Raw log: `logs/runtime_interface_map.log`

## 3. Device IDs

- LEFT_ARM: `rbd03ebf4ebf83c6a6a64754454bc520a`
- RIGHT_ARM: `r412d237980e3167577d7aece10f7aedb`
- LEFT_HAND: `r136d7b4b6e527ea3875679b4bf7eeb7d`
- RIGHT_HAND: `rcd72e2daf71f064c29aa45d4eeceeca9`

## 4. Runtime Namespace

Runtime namespace detected from live topics: `/gs_1eebee6f37512bbc1d125b25511e912c`.

## 5. Camera Interfaces

| Camera | Topic | Type | Resolution | Encoding | FPS | QoS | Frames | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_rgb | `/gs_1eebee6f37512bbc1d125b25511e912c/r6ef2dc_tp_cam_303d2b1ce0` | sensor_msgs/msg/Image | x |  |  | 1 / 2 / depth=6 | 0 | CHECK |
| left_wrist_rgb | `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_cam_069a6739f3` | sensor_msgs/msg/Image | x |  |  | 1 / 2 / depth=6 | 0 | CHECK |
| right_wrist_rgb | `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_cam_3c67aef2bc` | sensor_msgs/msg/Image | x |  |  | 1 / 2 / depth=6 | 0 | CHECK |

## 6. Depth Interfaces

| Topic | Type | Resolution | Encoding | FPS | Frames |
| --- | --- | --- | --- | --- | --- |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_dcam_a7a6349c11` | sensor_msgs/msg/Image | 320x180 | 32FC1 | 4.47 | 42 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r6ef2dc_tp_dcam_861ff01d8c` | sensor_msgs/msg/Image | x |  |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_dcam_3d5a0139cf` | sensor_msgs/msg/Image | 320x180 | 32FC1 | 4.21 | 42 |

## 7. Arm State Interfaces

### LEFT_ARM

| Topic | Category | Type | Hz | Frames |
| --- | --- | --- | --- | --- |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_cam_069a6739f3` | CAMERA_RGB | sensor_msgs/msg/Image |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_dcam_3d5a0139cf` | CAMERA_DEPTH | sensor_msgs/msg/Image | 4.21 | 42 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_dcam_3d5a0139cf/camera_info` | CAMERA_INFO | sensor_msgs/msg/CameraInfo |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_dcam_3d5a0139cf/points` | POINT_CLOUD | sensor_msgs/msg/PointCloud2 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_08fa69bc43` | ARM_JOINT_STATE | sensor_msgs/msg/JointState | 119.58 | 1188 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_1pa9vnh1mr` | ARM_JOINT_STATE | sensor_msgs/msg/JointState | 117.49 | 1163 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_2m4fzrdssg` | ARM_JOINT_STATE | sensor_msgs/msg/JointState | 113.51 | 1120 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_4jjp9rlwus` | ARM_JOINT_STATE | sensor_msgs/msg/JointState | 118.70 | 1175 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_rsqed0qcrb` | ARM_JOINT_STATE | sensor_msgs/msg/JointState | 115.31 | 1141 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_y1vmospyub` | ARM_JOINT_STATE | sensor_msgs/msg/JointState | 114.93 | 1134 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_y8sm0inqbu` | ARM_JOINT_STATE | sensor_msgs/msg/JointState | 117.02 | 1162 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_8042286a13` | FLOAT_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_bml0rwqx1y` | FLOAT_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_erahqe3xb9` | FLOAT_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_m1cl7k7oq3` | FLOAT_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_mwihphhu6b` | FLOAT_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_q2hhe71iur` | FLOAT_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_tpj019csu8` | FLOAT_SENSOR | std_msgs/msg/Float64 |  | 0 |

### RIGHT_ARM

| Topic | Category | Type | Hz | Frames |
| --- | --- | --- | --- | --- |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_cam_3c67aef2bc` | CAMERA_RGB | sensor_msgs/msg/Image |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_dcam_a7a6349c11` | CAMERA_DEPTH | sensor_msgs/msg/Image | 4.47 | 42 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_dcam_a7a6349c11/camera_info` | CAMERA_INFO | sensor_msgs/msg/CameraInfo |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_dcam_a7a6349c11/points` | POINT_CLOUD | sensor_msgs/msg/PointCloud2 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_08fa69bc43` | ARM_JOINT_STATE | sensor_msgs/msg/JointState | 114.62 | 1136 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_1pa9vnh1mr` | ARM_JOINT_STATE | sensor_msgs/msg/JointState | 117.86 | 1169 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_2m4fzrdssg` | ARM_JOINT_STATE | sensor_msgs/msg/JointState | 117.93 | 1168 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_4jjp9rlwus` | ARM_JOINT_STATE | sensor_msgs/msg/JointState | 115.50 | 1146 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_rsqed0qcrb` | ARM_JOINT_STATE | sensor_msgs/msg/JointState | 114.42 | 1132 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_y1vmospyub` | ARM_JOINT_STATE | sensor_msgs/msg/JointState | 117.03 | 1163 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_y8sm0inqbu` | ARM_JOINT_STATE | sensor_msgs/msg/JointState | 115.73 | 1147 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_8042286a13` | FLOAT_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_bml0rwqx1y` | FLOAT_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_erahqe3xb9` | FLOAT_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_m1cl7k7oq3` | FLOAT_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_mwihphhu6b` | FLOAT_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_q2hhe71iur` | FLOAT_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_tpj019csu8` | FLOAT_SENSOR | std_msgs/msg/Float64 |  | 0 |

## 8. Hand State Interfaces

### LEFT_HAND

| Topic | Category | Type | Hz | Frames |
| --- | --- | --- | --- | --- |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_094820dc35` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 119.33 | 1194 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_112a748ae9` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 120.33 | 1204 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_1f92f60733` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 118.73 | 1188 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_1fd37302bf` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 118.83 | 1189 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_2764cf5b12` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 119.83 | 1199 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_2f92dfedbc` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 120.33 | 1204 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_88eebd515e` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 118.70 | 1187 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_8e6c9ec41d` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 119.80 | 1198 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_b94282a9e8` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 119.70 | 1197 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_cc032bea74` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 119.90 | 1199 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_f3c793771c` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 118.40 | 1184 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_2we23l45vn` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_7umqt4htvd` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_9zf63en5i6` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_efaavyy3ew` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_ewjv7lz2a2` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_f4uo6e7q5d` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_j5j3ywk9yu` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_lyprfel66z` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_q1bckmw1rk` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_ts7vjc7cs0` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_v3dni6avfs` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_1ff4n99g6p` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 116.45 | 1155 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_26hns2y9iz` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 116.43 | 1157 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_3id78faqov` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 117.77 | 1170 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_c344tguind` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 117.42 | 1165 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_f1re3fuf56` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 116.13 | 1146 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_gww3yfamdt` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 117.69 | 1168 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_k3y7rwcnkq` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 114.97 | 1131 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_kfl2g3asap` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 118.46 | 1174 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_l2kojhzci0` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 116.44 | 1152 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_o6aptfg62g` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 115.51 | 1144 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_zp3f47zr30` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 117.44 | 1161 |

### RIGHT_HAND

| Topic | Category | Type | Hz | Frames |
| --- | --- | --- | --- | --- |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_0c06b0dccc` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 117.27 | 1157 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_0e0e800e4b` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 116.96 | 1154 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_22c267ca12` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 117.37 | 1158 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_2c1026959b` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 117.06 | 1155 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_3de69eda26` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 117.27 | 1157 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_5d58bffa87` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 116.96 | 1154 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_7d319548fa` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 116.76 | 1152 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_9955d0acad` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 116.86 | 1153 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_a9202465ea` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 116.66 | 1151 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_d2a6541a1e` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 116.97 | 1154 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_f918ef8abb` | FORCE_TORQUE | geometry_msgs/msg/Wrench | 116.66 | 1151 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_3l6z2qf3g3` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_6zh5gnlx1o` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_89dj1x750w` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_bfs2oeszoq` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_fn2waoflka` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_ja1sdnyb6f` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_mlp3n2l28s` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_n4szlwf6cw` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_oxmoqtj4qb` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_w7tllolabx` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_z0d2smrbqm` | POSITION_MOTOR_SENSOR | std_msgs/msg/Float64 |  | 0 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_3gpp28c61u` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 115.34 | 1138 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_4d9q0h9sn8` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 118.04 | 1168 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_8n8u1jtrj3` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 118.38 | 1175 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_cxjshiucra` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 117.26 | 1165 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_ilteetn3js` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 116.54 | 1148 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_msti8uutyd` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 117.41 | 1155 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_q2al5ws592` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 116.97 | 1155 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_up6h8dvvum` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 118.96 | 1179 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_v0b7dxyble` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 118.42 | 1170 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_wtszomzf2i` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 114.76 | 1140 |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_zrbc9dcp12` | HAND_JOINT_STATE | sensor_msgs/msg/JointState | 116.78 | 1155 |

## 9. JointState Mapping

| Topic | Device | Joint Names | Count | Hz | Meaning | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_08fa69bc43` | RIGHT_ARM | r412d23_joint_25af24fe79 | 1 | 114.62 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_1pa9vnh1mr` | RIGHT_ARM | r412d23_joint_d828c7277a | 1 | 117.86 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_2m4fzrdssg` | RIGHT_ARM | r412d23_joint_38783523af | 1 | 117.93 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_4jjp9rlwus` | RIGHT_ARM | r412d23_joint_a3a9be1d5d | 1 | 115.50 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_rsqed0qcrb` | RIGHT_ARM | r412d23_joint_90d3a309ba | 1 | 114.42 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_y1vmospyub` | RIGHT_ARM | r412d23_joint_2f2499504f | 1 | 117.03 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_y8sm0inqbu` | RIGHT_ARM | r412d23_joint_34d53cf32b | 1 | 115.73 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_08fa69bc43` | LEFT_ARM | rbd03eb_joint_5a89947698 | 1 | 119.58 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_1pa9vnh1mr` | LEFT_ARM | rbd03eb_joint_7572c2386d | 1 | 117.49 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_2m4fzrdssg` | LEFT_ARM | rbd03eb_joint_164adb5dca | 1 | 113.51 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_4jjp9rlwus` | LEFT_ARM | rbd03eb_joint_a6155f358a | 1 | 118.70 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_rsqed0qcrb` | LEFT_ARM | rbd03eb_joint_0e4e3b51ac | 1 | 115.31 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_y1vmospyub` | LEFT_ARM | rbd03eb_joint_f555168960 | 1 | 114.93 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_y8sm0inqbu` | LEFT_ARM | rbd03eb_joint_625db43b33 | 1 | 117.02 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_1ff4n99g6p` | LEFT_HAND | r136d7b_joint_623071b675 | 1 | 116.45 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_26hns2y9iz` | LEFT_HAND | r136d7b_joint_4851a0ceaa | 1 | 116.43 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_3id78faqov` | LEFT_HAND | r136d7b_joint_02d7e3c450 | 1 | 117.77 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_c344tguind` | LEFT_HAND | r136d7b_joint_7c9b0f5dac | 1 | 117.42 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_f1re3fuf56` | LEFT_HAND | r136d7b_joint_9f052fb9b2 | 1 | 116.13 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_gww3yfamdt` | LEFT_HAND | r136d7b_joint_c2358099a4 | 1 | 117.69 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_k3y7rwcnkq` | LEFT_HAND | r136d7b_joint_1c463fc81a | 1 | 114.97 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_kfl2g3asap` | LEFT_HAND | r136d7b_joint_20ac046af7 | 1 | 118.46 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_l2kojhzci0` | LEFT_HAND | r136d7b_joint_4e88f26ae9 | 1 | 116.44 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_o6aptfg62g` | LEFT_HAND | r136d7b_joint_cab9143c1f | 1 | 115.51 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_zp3f47zr30` | LEFT_HAND | r136d7b_joint_88af9e2c20 | 1 | 117.44 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_3gpp28c61u` | RIGHT_HAND | rcd72e2_joint_a128a8a7e8 | 1 | 115.34 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_4d9q0h9sn8` | RIGHT_HAND | rcd72e2_joint_9efa81f107 | 1 | 118.04 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_8n8u1jtrj3` | RIGHT_HAND | rcd72e2_joint_a80961cc30 | 1 | 118.38 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_cxjshiucra` | RIGHT_HAND | rcd72e2_joint_ce3fcf854f | 1 | 117.26 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_ilteetn3js` | RIGHT_HAND | rcd72e2_joint_4283bce291 | 1 | 116.54 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_msti8uutyd` | RIGHT_HAND | rcd72e2_joint_d6fdfa63a1 | 1 | 117.41 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_q2al5ws592` | RIGHT_HAND | rcd72e2_joint_31658f988d | 1 | 116.97 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_up6h8dvvum` | RIGHT_HAND | rcd72e2_joint_2c7995baf1 | 1 | 118.96 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_v0b7dxyble` | RIGHT_HAND | rcd72e2_joint_d4c3ace083 | 1 | 118.42 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_wtszomzf2i` | RIGHT_HAND | rcd72e2_joint_b812826a3b | 1 | 114.76 | per-sensor/per-joint or hand state | INFERRED |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_zrbc9dcp12` | RIGHT_HAND | rcd72e2_joint_2fca2e44d0 | 1 | 116.78 | per-sensor/per-joint or hand state | INFERRED |

## 10. Force/Torque Mapping

| Topic | Device | Frame | Hz | Sample | Meaning/Confidence |
| --- | --- | --- | --- | --- | --- |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_094820dc35` | LEFT_HAND |  | 119.33 | F={'x': 0.17707168078923888, 'y': -0.007047860620031549, 'z': 0.08308125419732472} T={'x': 0.00028451265961848717, 'y': 0.007134971479688523, 'z': -1.117584569637112e-06} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_112a748ae9` | LEFT_HAND |  | 120.33 | F={'x': 0.0015360275389430585, 'y': -0.003216811058577884, 'z': 0.03676851438961362} T={'x': 6.271801264987323e-05, 'y': -0.00021244041732775107, 'z': -2.120611320067502e-05} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_1f92f60733` | LEFT_HAND |  | 118.73 | F={'x': 0.008702925634559069, 'y': -7.954533039442857e-11, 'z': 0.20912122278209605} T={'x': -0.008362935601931281, 'y': -0.004735125550656013, 'z': 0.0003480373980473338} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_1fd37302bf` | LEFT_HAND |  | 118.83 | F={'x': 0.0015360467640784835, 'y': -1.4039454495224002e-11, 'z': 0.03690942460641043} T={'x': 1.8855683869347417e-07, 'y': -0.00021336840888940223, 'z': -7.847185931086263e-09} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_2764cf5b12` | LEFT_HAND |  | 119.83 | F={'x': 0.0015360337207399478, 'y': -0.0019316781551968515, 'z': 0.03685852829875057} T={'x': 3.773712014476133e-05, 'y': -0.00021303321089659558, 'z': -1.2737271685974504e-05} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_2f92dfedbc` | LEFT_HAND |  | 120.33 | F={'x': 0.0025677024504580123, 'y': 0.003229079279656418, 'z': 0.06161435921421192} T={'x': -0.00013045700700992324, 'y': 0.00010626514958199856, 'z': -1.3249859594642968e-07} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_88eebd515e` | LEFT_HAND |  | 118.70 | F={'x': 0.0015360398232006048, 'y': 0.001931685801455152, 'z': 0.036858674734216536} T={'x': -3.7361335321687135e-05, 'y': -0.00021303410223592732, 'z': 1.2721657867466903e-05} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_8e6c9ec41d` | LEFT_HAND |  | 119.80 | F={'x': 0.06767880698656228, 'y': -0.002693772355047188, 'z': 0.03175459871387671} T={'x': 6.198560172247008e-05, 'y': 0.0016887611441404856, 'z': 1.1148826430857715e-05} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_b94282a9e8` | LEFT_HAND |  | 119.70 | F={'x': 0.002567727202000307, 'y': -0.003229110453460988, 'z': 0.06161495314759097} T={'x': 0.0001309611292829839, 'y': 0.00010626994057547247, 'z': 1.117410885608875e-07} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_cc032bea74` | LEFT_HAND |  | 119.90 | F={'x': 0.002567736380629314, 'y': -0.005377457483759432, 'z': 0.061464947511883763} T={'x': 0.00021792181266835897, 'y': 0.00010626626451786821, 'z': 1.9322481125104337e-07} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_f3c793771c` | LEFT_HAND |  | 118.40 | F={'x': 0.002567720488413585, 'y': -2.3468941200859147e-11, 'z': 0.061699349259383425} T={'x': 2.522473390105572e-07, 'y': 0.00010627027433086908, 'z': -1.0497649882032233e-08} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_0c06b0dccc` | RIGHT_HAND |  | 117.27 | F={'x': 0.06839049375644941, 'y': -0.00010317470342599632, 'z': 0.030303738131835484} T={'x': 2.3602373833577546e-06, 'y': 0.001699009722963378, 'z': 4.579311038784194e-07} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_0e0e800e4b` | RIGHT_HAND |  | 116.96 | F={'x': -9.832437786218885e-05, 'y': 2.360611532632159e-11, 'z': 0.06173617392231237} T={'x': -2.9186525521076993e-07, 'y': -1.6070329520417434e-06, 'z': -4.6483981560767647e-10} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_22c267ca12` | RIGHT_HAND |  | 117.37 | F={'x': -9.832331023482309e-05, 'y': 0.003230994185982585, 'z': 0.061650896818497086} T={'x': -0.00013107397409673602, 'y': -1.6125448666852038e-06, 'z': -1.2453190993363493e-07} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_2c1026959b` | RIGHT_HAND |  | 117.06 | F={'x': -5.8808207228213806e-05, 'y': 0.0032181787140108833, 'z': 0.03678414684511934} T={'x': -6.27402341309261e-05, 'y': -0.00024352905551134536, 'z': 2.120561298550182e-05} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_3de69eda26` | RIGHT_HAND |  | 117.27 | F={'x': -0.000333341397681444, 'y': 8.002975539195877e-11, 'z': 0.20929929017021776} T={'x': 0.008369841718790216, 'y': -0.004833888857515221, 'z': 1.3330265580451193e-05} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_5d58bffa87` | RIGHT_HAND |  | 116.96 | F={'x': -9.832498673607599e-05, 'y': -0.003231049230169002, 'z': 0.061651948023776486} T={'x': 0.00013049254797574132, 'y': -1.6085662822762569e-06, 'z': 1.238131388468814e-07} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_7d319548fa` | RIGHT_HAND |  | 116.76 | F={'x': -5.88089032185859e-05, 'y': -0.00193251449052316, 'z': 0.03687448702807204} T={'x': 3.740506453139209e-05, 'y': -0.0002441241950763579, 'z': -1.2734381722923296e-05} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_9955d0acad` | RIGHT_HAND |  | 116.86 | F={'x': -5.880840758366956e-05, 'y': 0.0019324982316988061, 'z': 0.03687417625239693} T={'x': -3.774290865966528e-05, 'y': -0.00024412223897218359, 'z': 1.2733740587512802e-05} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_a9202465ea` | RIGHT_HAND |  | 116.66 | F={'x': -5.880876753337627e-05, 'y': 1.4119047510973709e-11, 'z': 0.03692500662539045} T={'x': -1.694334131671361e-07, 'y': -0.0002444571967449752, 'z': -2.6975536675287775e-10} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_d2a6541a1e` | RIGHT_HAND |  | 116.97 | F={'x': 0.17893811971508103, 'y': -0.0002699481524279322, 'z': 0.07928724627902091} T={'x': 1.002683476876714e-05, 'y': 0.007209454159853478, 'z': 1.917027988515817e-06} | UNKNOWN link/finger unless frame/topic identifies it |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_f918ef8abb` | RIGHT_HAND |  | 116.66 | F={'x': -9.832255376663118e-05, 'y': 0.005380533845088239, 'z': 0.06150011066855583} T={'x': -0.00021808198627078183, 'y': -1.6202184761507408e-06, 'z': -2.0690592805358663e-07} | UNKNOWN link/finger unless frame/topic identifies it |

## 11. PM / SM Mapping

| Topic | Device | Suffix | Type | Sample Range | Hz |
| --- | --- | --- | --- | --- | --- |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_2we23l45vn` | LEFT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_7umqt4htvd` | LEFT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_9zf63en5i6` | LEFT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_efaavyy3ew` | LEFT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_ewjv7lz2a2` | LEFT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_f4uo6e7q5d` | LEFT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_j5j3ywk9yu` | LEFT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_lyprfel66z` | LEFT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_q1bckmw1rk` | LEFT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_ts7vjc7cs0` | LEFT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_v3dni6avfs` | LEFT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_3l6z2qf3g3` | RIGHT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_6zh5gnlx1o` | RIGHT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_89dj1x750w` | RIGHT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_bfs2oeszoq` | RIGHT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_fn2waoflka` | RIGHT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_ja1sdnyb6f` | RIGHT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_mlp3n2l28s` | RIGHT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_n4szlwf6cw` | RIGHT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_oxmoqtj4qb` | RIGHT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_w7tllolabx` | RIGHT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_z0d2smrbqm` | RIGHT_HAND | pm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_8042286a13` | RIGHT_ARM | sm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_bml0rwqx1y` | RIGHT_ARM | sm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_erahqe3xb9` | RIGHT_ARM | sm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_m1cl7k7oq3` | RIGHT_ARM | sm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_mwihphhu6b` | RIGHT_ARM | sm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_q2hhe71iur` | RIGHT_ARM | sm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_tpj019csu8` | RIGHT_ARM | sm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_8042286a13` | LEFT_ARM | sm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_bml0rwqx1y` | LEFT_ARM | sm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_erahqe3xb9` | LEFT_ARM | sm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_m1cl7k7oq3` | LEFT_ARM | sm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_mwihphhu6b` | LEFT_ARM | sm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_q2hhe71iur` | LEFT_ARM | sm | std_msgs/msg/Float64 | NO_SAMPLES |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_tpj019csu8` | LEFT_ARM | sm | std_msgs/msg/Float64 | NO_SAMPLES |  |

- PS: Position sensor or per-joint state publisher (INFERRED). Evidence: Live _tp_ps_ topics use types ['sensor_msgs/msg/JointState']; JointState content is per topic, not assumed complete robot state.
- PM: Position/motor scalar sensor (INFERRED). Evidence: Live _tp_pm_ topics are Float64 with sampled scalar range NO_SAMPLES.
- SM: Scalar sensor/motor telemetry (UNKNOWN). Evidence: Live _tp_sm_ topics are Float64 with sampled scalar range NO_SAMPLES; no local manual/package definition found.

## 12. SDK vs ROS State

| Device | Status | ROS Topic | Max Abs Error |
| --- | --- | --- | --- |
| LEFT_ARM | NOT_MAPPED |  |  |
| RIGHT_ARM | NOT_MAPPED |  |  |
| LEFT_HAND | NOT_MAPPED |  |  |
| RIGHT_HAND | NOT_MAPPED |  |  |

## 13. Timestamp / Synchronization

Time domain: SAME TIME DOMAIN. Max sampled ROS stamp delta: 0.027000000000001023.

```json
{
  "left_arm": {
    "topic": "/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_08fa69bc43",
    "ros_stamp": {
      "sec": 18,
      "nanosec": 849000000,
      "float": 18.849
    },
    "wall_time": 1786689273.5476785
  },
  "right_arm": {
    "topic": "/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_08fa69bc43",
    "ros_stamp": {
      "sec": 18,
      "nanosec": 873000000,
      "float": 18.873
    },
    "wall_time": 1786689273.5711877
  },
  "left_hand": {
    "topic": "/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_1ff4n99g6p",
    "ros_stamp": {
      "sec": 18,
      "nanosec": 867000000,
      "float": 18.867
    },
    "wall_time": 1786689273.5632665
  },
  "right_hand": {
    "topic": "/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_3gpp28c61u",
    "ros_stamp": {
      "sec": 18,
      "nanosec": 876000000,
      "float": 18.876
    },
    "wall_time": 1786689273.5792863
  }
}
```

## 14. Fixed RGB Investigation

Fixed RGB result: TOPIC EXISTS BUT NO DATA. Publisher count: 1. Frames: 0. QoS: 1 / 2 / depth=6.
If the topic exists with publishers but frames remain zero while wrist cameras work, the most likely cause is either publisher inactivity in the current scene or QoS/subscriber compatibility. This run used a BEST_EFFORT/VOLATILE subscriber.

## 15. Recommended ACT Observation Sources

- fixed_rgb -> ROS topic /gs_1eebee6f37512bbc1d125b25511e912c/r6ef2dc_tp_cam_303d2b1ce0
- left_wrist_rgb -> ROS topic /gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_cam_069a6739f3
- right_wrist_rgb -> ROS topic /gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_cam_3c67aef2bc
- left_arm_qpos -> SDK get_joint_angles()
- right_arm_qpos -> SDK get_joint_angles()
- left_hand_qpos -> SDK get_joint_angles()
- right_hand_qpos -> SDK get_joint_angles()

## 16. Recommended Recorder Architecture

```text
ROS Camera
        \
Robot State -> Synchronizer -> Episode Recorder
        /
Action
```

Recommended first Recorder: SDK State + ROS Camera, because SDK qpos is explicit and already validated while ROS JointState mapping still has CHECK/UNKNOWN items.

## 17. Interfaces Useful for Expert

| Interface | Use | Reason |
| --- | --- | --- |
| RGB cameras | ACT INPUT | Direct visual observation. |
| Robot qpos | ACT INPUT | Policy proprioception. |
| SDK pose/state | EXPERT ONLY | Read-only debugging and state sanity checks. |
| FTS | DEBUG / SUCCESS CHECK | Contact/force signal; mapping to fingers may still be unknown. |
| Depth/PointCloud | OPTIONAL | 3D perception/debug; not required for first ACT observation. |

## 18. Interfaces Not Needed Initially

- Depth images and PointCloud2 can be ignored for first recorder unless 3D perception/debug is needed.
- FTS can be ignored for first ACT input; keep for contact diagnostics or success heuristics.
- Unknown Float64 PM/SM topics should not be used as policy input until semantics are confirmed.

## 19. Confirmed Facts

- Runtime namespace is discovered dynamically from `/gs_<id>/...` topics.
- `_tp_cam_`, `_tp_cami_`, `_tp_rgbd_`, and `_tp_fts_` meanings are documented in local Rabo manual snapshots.
- All listed live message types and sampled frequencies are from read-only ROS discovery/subscription.

## 20. Inferred Facts

- `_tp_ps_` is treated as per-position-sensor or per-joint state unless a full-device JointState matches SDK qpos.
- Complete A7 state candidate requires JointState position length 7 and SDK comparison.
- O6 hand state candidate requires matching SDK hand joint vector length/order.

## 21. Remaining Unknowns

- ps meaning is INFERRED
- pm meaning is INFERRED
- sm meaning is UNKNOWN
- fixed_rgb did not produce frames during sampling
- left_wrist_rgb did not produce frames during sampling
- right_wrist_rgb did not produce frames during sampling

## 22. Final Conclusion

- RUNTIME INTERFACE MAPPING: PASS
- CAMERA OBSERVATION: CHECK
- ROBOT STATE OBSERVATION: CHECK
- READY FOR RECORDER: NO
- READY FOR THREE-NUT EXPERT: NO

## PointCloud Interfaces

| Topic | Device | Size | Fields | Point Step | Hz |
| --- | --- | --- | --- | --- | --- |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_dcam_a7a6349c11/points` | RIGHT_ARM | x |  |  |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r6ef2dc_tp_dcam_861ff01d8c/points` | None | x |  |  |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_dcam_3d5a0139cf/points` | LEFT_ARM | x |  |  |  |
