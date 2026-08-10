"""右臂末端位姿控制 + 右手力控握紧。"""

import time
from rabo_robocap import LinkerArmA7, LinkerHandO6Right

# 场景中右臂和右手的机器人 ID
RIGHT_ARM_ID = "r412d237980e3167577d7aece10f7aedb"
RIGHT_HAND_ID = "rcd72e2daf71f064c29aa45d4eeceeca9"


def run():
    # ===== 右臂控制 =====
    arm = LinkerArmA7(robot_id=RIGHT_ARM_ID, mode="sim")

    # 右臂末端位姿: 位置 (0.4, 0.1, -0.3) m, 姿态 RPY=(-1.57, 0, 1.57) rad
    print("右臂移动中...")
    arm.move_joints([0,-1.57,0,0,0,0,0])
    arm.move_joints([1.57,-1.57,0,0,0,0,0])
    ok = arm.move_to(-0.43, 0.16, -0.2, roll=0, pitch=0.8, yaw=0)
    ok = arm.move_to(-0.43, 0.16, -0.33, roll=0, pitch=0.8, yaw=0)

    if ok:
        print("右臂已到达目标位姿!")
    else:
        print("移动失败, 请检查目标位姿是否可达。")

    # ===== 右手控制 =====
    hand = LinkerHandO6Right(robot_id=RIGHT_HAND_ID, mode="sim")

    # 1) 大拇指第一个关节 (thumb_cmc_yaw) 内扣
    print("大拇指内扣...")
    hand.clench(thumb_rotation=1.0)

    # 2) 力控握紧
    print("力控握紧...")
    hand.grasp_force(strength=0.5)

    ok = arm.move_to(-0.4, 0.12, -0.2, roll=0, pitch=0.8, yaw=0)
    ok = arm.move_to(-0.4, 0, -0.2, roll=0, pitch=0.8, yaw=0)

    arm.shutdown()
    hand.shutdown()
