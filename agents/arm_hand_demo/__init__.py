"""右臂末端位姿控制 + 右手力控握紧。"""

import time
from rabo_robocap import LinkerArmA7, LinkerHandO6Right,LinkerHandO6Left
# 场景中右臂和右手的机器人 ID
RIGHT_ARM_ID = "r412d237980e3167577d7aece10f7aedb"
RIGHT_HAND_ID = "rcd72e2daf71f064c29aa45d4eeceeca9"

LEFT_ARM_ID = "rbd03ebf4ebf83c6a6a64754454bc520a"
LEFT_HAND_ID = "r136d7b4b6e527ea3875679b4bf7eeb7d"


def run():
    # ===== 右臂控制 =====
    right_arm = LinkerArmA7(robot_id=RIGHT_ARM_ID, mode="sim")
    left_arm = LinkerArmA7(robot_id=LEFT_ARM_ID, mode="sim")

    # 右臂末端位姿: 位置 (0.4, 0.1, -0.3) m, 姿态 RPY=(-1.57, 0, 1.57) rad
    print("右臂移动中...")
    right_arm.move_joints([0,-1.57,0,0,0,0,0])
    right_arm.move_joints([1.57,-1,0,0,0,0,0])

    left_arm.move_joints([0,-1.57,0,0,0,0,0])
    left_arm.move_joints([-1.57,-1,0,0,0,0,0])

    ok = right_arm.move_to(-0.43, 0.15, -0.2, roll=0, pitch=0.8, yaw=0)
    ok = right_arm.move_to(-0.43, 0.15, -0.33, roll=0, pitch=0.8, yaw=0)

    if ok:
        print("右臂已到达目标位姿!")
    else:
        print("移动失败, 请检查目标位姿是否可达。")

    # ===== 右手控制 =====
    right_hand = LinkerHandO6Right(robot_id=RIGHT_HAND_ID, mode="sim")
    left_hand = LinkerHandO6Left(robot_id=LEFT_HAND_ID, mode="sim")

    # 1) 大拇指第一个关节 (thumb_cmc_yaw) 内扣
    print("大拇指内扣...")
    right_hand.clench(thumb_rotation=1.0)

    # 2) 力控握紧
    print("力控握紧...")
    right_hand.grasp_force(strength=0.5,fingers=[1, 3,4])

    ok = right_arm.move_to(-0.4, 0.12, -0.03, roll=0, pitch=0.8, yaw=0)
    ok = right_arm.move_to(-0.4, 0, -0.03, roll=0, pitch=0.8, yaw=0)

    left_arm.move_to(0.43, 0.3, -0.1, roll=0, pitch=1.3, yaw=1.57)
    left_arm.move_to(0.43, 0.1, -0.2, roll=0, pitch=1.3, yaw=1.57)
    left_hand.clench(0.3, 0, 0.3,0.3,0.3,0.3) 

    right_hand.clench(0,0,0,0,0,0) 

    left_hand.clench(1, 0, 0.3,0.3,0.3,0.3) 
    left_hand.grasp_force(strength=0.5)

    left_arm.move_to(0.48, 0.25, -0.2, roll=0, pitch=0, yaw=0)
    left_hand.clench(0,0,0,0,0,0) 

    right_arm.shutdown()
    left_arm.shutdown()
    right_hand.shutdown()
    left_hand.shutdown()
