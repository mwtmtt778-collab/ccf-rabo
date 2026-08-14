# RABO Camera Test Report

- Generated: 2026-08-14 12:08:23 +0800
- Duration: 10.0s
- Result: CHECK
- Output directory: `outputs`
- Raw log: `logs/camera_test.log`

## Read-only constraints

- No arm motion APIs were called.
- No hand actuation APIs were called.
- No object/entity pose or simulation reset APIs were called.
- No camera parameters were modified.

## Known RGB topics

| Camera | Expected topic | Discovery | ROS type |
| --- | --- | --- | --- |
| fixed_rgb | `r6ef2dc_tp_cam_303d2b1ce0` | MISSING |  |
| left_wrist_rgb | `rbd03eb_tp_cam_069a6739f3` | MISSING |  |
| right_wrist_rgb | `r412d23_tp_cam_3c67aef2bc` | MISSING |  |

## Sampled topics

| Topic | Type | Frames | Approx FPS | Sample | Error |
| --- | --- | ---: | ---: | --- | --- |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_cam_3c67aef2bc` | sensor_msgs/msg/Image | 33 | 3.53 | outputs/gs_1eebee6f37512bbc1d125b25511e912c__r412d23_tp_cam_3c67aef2bc.ppm |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_dcam_a7a6349c11/camera_info` | sensor_msgs/msg/CameraInfo | 0 |  |  | unsupported topic types: ['sensor_msgs/msg/CameraInfo'] |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r6ef2dc_tp_cam_303d2b1ce0` | sensor_msgs/msg/Image | 0 |  |  |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/r6ef2dc_tp_dcam_861ff01d8c/camera_info` | sensor_msgs/msg/CameraInfo | 0 |  |  | unsupported topic types: ['sensor_msgs/msg/CameraInfo'] |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_cam_069a6739f3` | sensor_msgs/msg/Image | 40 | 4.19 | outputs/gs_1eebee6f37512bbc1d125b25511e912c__rbd03eb_tp_cam_069a6739f3.ppm |  |
| `/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_dcam_3d5a0139cf/camera_info` | sensor_msgs/msg/CameraInfo | 0 |  |  | unsupported topic types: ['sensor_msgs/msg/CameraInfo'] |

## Discovery command

```text
$ ros2 topic list -t --no-daemon
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_094820dc35 [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_112a748ae9 [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_1f92f60733 [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_1fd37302bf [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_2764cf5b12 [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_2f92dfedbc [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_88eebd515e [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_8e6c9ec41d [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_b94282a9e8 [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_cc032bea74 [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_fts_f3c793771c [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_2we23l45vn [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_7umqt4htvd [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_9zf63en5i6 [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_efaavyy3ew [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_ewjv7lz2a2 [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_f4uo6e7q5d [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_j5j3ywk9yu [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_lyprfel66z [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_q1bckmw1rk [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_ts7vjc7cs0 [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_pm_v3dni6avfs [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_1ff4n99g6p [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_26hns2y9iz [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_3id78faqov [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_c344tguind [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_f1re3fuf56 [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_gww3yfamdt [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_k3y7rwcnkq [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_kfl2g3asap [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_l2kojhzci0 [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_o6aptfg62g [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r136d7b_tp_ps_zp3f47zr30 [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_cam_3c67aef2bc [sensor_msgs/msg/Image]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_dcam_a7a6349c11 [sensor_msgs/msg/Image]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_dcam_a7a6349c11/camera_info [sensor_msgs/msg/CameraInfo]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_dcam_a7a6349c11/points [sensor_msgs/msg/PointCloud2]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_08fa69bc43 [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_1pa9vnh1mr [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_2m4fzrdssg [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_4jjp9rlwus [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_rsqed0qcrb [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_y1vmospyub [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_ps_y8sm0inqbu [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_8042286a13 [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_bml0rwqx1y [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_erahqe3xb9 [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_m1cl7k7oq3 [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_mwihphhu6b [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_q2hhe71iur [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_sm_tpj019csu8 [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/r6ef2dc_tp_cam_303d2b1ce0 [sensor_msgs/msg/Image]
/gs_1eebee6f37512bbc1d125b25511e912c/r6ef2dc_tp_dcam_861ff01d8c [sensor_msgs/msg/Image]
/gs_1eebee6f37512bbc1d125b25511e912c/r6ef2dc_tp_dcam_861ff01d8c/camera_info [sensor_msgs/msg/CameraInfo]
/gs_1eebee6f37512bbc1d125b25511e912c/r6ef2dc_tp_dcam_861ff01d8c/points [sensor_msgs/msg/PointCloud2]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_cam_069a6739f3 [sensor_msgs/msg/Image]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_dcam_3d5a0139cf [sensor_msgs/msg/Image]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_dcam_3d5a0139cf/camera_info [sensor_msgs/msg/CameraInfo]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_dcam_3d5a0139cf/points [sensor_msgs/msg/PointCloud2]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_08fa69bc43 [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_1pa9vnh1mr [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_2m4fzrdssg [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_4jjp9rlwus [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_rsqed0qcrb [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_y1vmospyub [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_ps_y8sm0inqbu [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_8042286a13 [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_bml0rwqx1y [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_erahqe3xb9 [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_m1cl7k7oq3 [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_mwihphhu6b [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_q2hhe71iur [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_sm_tpj019csu8 [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_0c06b0dccc [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_0e0e800e4b [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_22c267ca12 [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_2c1026959b [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_3de69eda26 [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_5d58bffa87 [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_7d319548fa [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_9955d0acad [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_a9202465ea [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_d2a6541a1e [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_fts_f918ef8abb [geometry_msgs/msg/Wrench]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_3l6z2qf3g3 [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_6zh5gnlx1o [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_89dj1x750w [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_bfs2oeszoq [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_fn2waoflka [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_ja1sdnyb6f [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_mlp3n2l28s [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_n4szlwf6cw [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_oxmoqtj4qb [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_w7tllolabx [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_pm_z0d2smrbqm [std_msgs/msg/Float64]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_3gpp28c61u [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_4d9q0h9sn8 [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_8n8u1jtrj3 [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_cxjshiucra [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_ilteetn3js [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_msti8uutyd [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_q2al5ws592 [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_up6h8dvvum [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_v0b7dxyble [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_wtszomzf2i [sensor_msgs/msg/JointState]
/gs_1eebee6f37512bbc1d125b25511e912c/rcd72e2_tp_ps_zrbc9dcp12 [sensor_msgs/msg/JointState]
/parameter_events [rcl_interfaces/msg/ParameterEvent]
/rosout [rcl_interfaces/msg/Log]
```
