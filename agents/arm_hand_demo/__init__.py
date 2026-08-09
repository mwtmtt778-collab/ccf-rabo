"""右臂末端位姿控制。"""

import time
from rabo_robocap import LinkerArmA7

# 场景中右臂的机器人 ID
RIGHT_ARM_ID = "r412d237980e3167577d7aece10f7aedb"


def run():
    # 初始化右臂
    arm = LinkerArmA7(robot_id=RIGHT_ARM_ID, mode="sim")

    # 右臂末端位姿: 位置 (0.4, 0.1, -0.3) m, 姿态 RPY=(-1.57, 0, 1.57) rad
    print("右臂移动中...")
    ok = arm.move_to(0.4, 0.1, -0.3, roll=-1.57, pitch=0.0, yaw=1.57)

    if ok:
        print("右臂已到达目标位姿!")
    else:
        print("移动失败, 请检查目标位姿是否可达。")

    arm.shutdown()
