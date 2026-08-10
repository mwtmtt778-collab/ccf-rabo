"""双臂协作抓取螺母: 右臂抓螺母B → 左臂辅助接取。"""

import random
import threading
from rabo_robocap import LinkerArmA7, LinkerHandO6Right, LinkerHandO6Left
from rabo_dev_kit import SetEntityPose

WORLD_ID = "ww4b1e6f8392584c89902a6783a9f53075"
NUT_A_ID = "thing_3eb64241-4166-4c4e-9144-edf733c885f1"
NUT_B_ID = "thing_e937f633-faed-4b2e-b609-a7b80825a64a"
NUT_C_ID = "thing_4a38ffae-a85b-43c9-8b20-abb0cf6dcac7"

# ── 机器人 ID ──
RIGHT_ARM_ID = "r412d237980e3167577d7aece10f7aedb"
RIGHT_HAND_ID = "rcd72e2daf71f064c29aa45d4eeceeca9"
LEFT_ARM_ID = "rbd03ebf4ebf83c6a6a64754454bc520a"
LEFT_HAND_ID = "r136d7b4b6e527ea3875679b4bf7eeb7d"


def run():
    # ══════════════════════════════════════════════════
    # 0. 初始化
    # ══════════════════════════════════════════════════

    # 右臂基座在桌面坐标系中的参考位置 (x, y)
    # right_arm_base = [-0.8316, -0.004]   # 备选位置
    right_arm_base = [-0.6816, -0.004]

    # 螺母 B 随机扰动位姿 [x, y, z, roll, pitch, yaw]
    nut_b_position = [
        -0.3213 + random.uniform(-0.02, 0.02),
        -0.1910 + random.uniform(-0.02, 0.02),
        0.2806, 0, 0, 0.5233,
    ]

    pose_setter = SetEntityPose(world=WORLD_ID)
    pose_setter.set(
        NUT_B_ID, (
            nut_b_position[0], nut_b_position[1], nut_b_position[2],
            nut_b_position[3], nut_b_position[4], nut_b_position[5],
        )
    )

    right_arm = LinkerArmA7(robot_id=RIGHT_ARM_ID, mode="sim")
    left_arm = LinkerArmA7(robot_id=LEFT_ARM_ID, mode="sim")
    right_hand = LinkerHandO6Right(robot_id=RIGHT_HAND_ID, mode="sim")
    left_hand = LinkerHandO6Left(robot_id=LEFT_HAND_ID, mode="sim")

    # ══════════════════════════════════════════════════
    # 1. 双臂预摆位: 关节空间粗定位（左右并行）
    # ══════════════════════════════════════════════════
    print("双臂预摆位（并行）...")

    def _right_pre_position():
        # 先收拢再展开到抓取预备姿态
        right_arm.move_joints([-1.57, -1.5, 0, -1.57, 0, -1, 0])
        right_arm.move_joints([0, 0, 0, -2, 0, 1, 0])

    def _left_pre_position():
        # 先摆到竖直再前倾到预备姿态
        left_arm.move_joints([0, -1.57, 0, 0, 0, 0, 0])
        left_arm.move_joints([-1.57, -0.7, 0, 0, 0, 0, 0])

    t_rp = threading.Thread(target=_right_pre_position)
    t_lp = threading.Thread(target=_left_pre_position)
    t_rp.start()
    t_lp.start()
    t_rp.join()
    t_lp.join()

    # ══════════════════════════════════════════════════
    # 2. 右臂笛卡尔逼近螺母 B
    # ══════════════════════════════════════════════════
    print("右臂逼近螺母...")
    # 根据螺母实际位姿计算末端目标 (带经验偏移)
    target_x = right_arm_base[0] - nut_b_position[0] + 0.06
    target_y = right_arm_base[1] - nut_b_position[1] - 0.01
    right_arm.move_to(target_x, target_y, -0.33, roll=0, pitch=0.8, yaw=0)

    # ══════════════════════════════════════════════════
    # 3. 右手抓取螺母
    # ══════════════════════════════════════════════════
    # 拇指第一关节 (thumb_cmc_yaw) 内扣，避免与物体干涉
    print("大拇指内扣...")
    right_hand.clench(thumb_rotation=1.0)

    # 力控握紧: 拇指弯曲 + 中指 + 小指施力抓取
    #   fingers 索引: 0=拇指旋转, 1=拇指弯曲, 2=食指, 3=中指, 4=无名指, 5=小指
    print("力控握紧...")
    right_hand.grasp_force(strength=1, fingers=[1, 3, 4])

    # ══════════════════════════════════════════════════
    # 4+5. 右臂提起移开 ‖ 左臂到交接位置（并行）
    # ══════════════════════════════════════════════════
    print("右臂提起 + 左臂接取（并行）...")

    def _right_lift():
        right_arm.move_to(-0.4, 0.12, -0.03, roll=0, pitch=0.8, yaw=0)
        right_arm.move_to(-0.4, 0, -0.03, roll=0, pitch=0.8, yaw=0)

    def _left_approach():
        left_arm.move_to(0.43, 0.3, -0.1, roll=0, pitch=1.3, yaw=1.57)
        left_arm.move_to(0.44, 0.1, -0.2, roll=0, pitch=1.3, yaw=1.57)
        # 左手指微张预备
        left_hand.clench(0.3, 0, 0.3, 0.3, 0.3, 0.3)

    t_rl = threading.Thread(target=_right_lift)
    t_la = threading.Thread(target=_left_approach)
    t_rl.start()
    t_la.start()
    t_rl.join()
    t_la.join()

    # ══════════════════════════════════════════════════
    # 6. 右手释放 → 左手抓取
    # ══════════════════════════════════════════════════
    print("交接: 右手释放, 左手抓取...")
    right_hand.clench(0, 0, 0, 0, 0, 0)            # 右手完全张开释放

    left_hand.clench(1, 0, 0.3, 0.3, 0.3, 0.3)    # 拇指内扣, 四指微合
    left_hand.grasp_force(strength=0.5)

    # ══════════════════════════════════════════════════
    # 7. 左臂移开 → 左手释放
    # ══════════════════════════════════════════════════
    print("左臂移开, 释放...")
    left_arm.move_to(0.43, 0.25, -0.2, roll=0, pitch=0, yaw=0)
    left_hand.clench(0, 0, 0, 0, 0, 0)             # 左手张开释放

    # ══════════════════════════════════════════════════
    # 8. 清理
    # ══════════════════════════════════════════════════
    right_arm.shutdown()
    left_arm.shutdown()
    right_hand.shutdown()
    left_hand.shutdown()
