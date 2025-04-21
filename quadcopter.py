# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate a quadcopter.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/quadcopter.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

# 添加命令行参数解析器
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
# 添加AppLauncher的命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析命令行参数
args_cli = parser.parse_args()

# 启动Omniverse应用程序
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort:skip


def main():
    """主函数."""
    # 加载仿真配置
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # 设置主摄像机视角
    sim.set_camera_view(eye=[0.5, 0.5, 1.0], target=[0.0, 0.0, 0.5])

    # 在场景中生成物体
    # 地面平面
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # 灯光
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # 机器人
    robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)

    # 创建机器人控制句柄
    robot = Articulation(robot_cfg)

    # 启动仿真器
    sim.reset()

    # 获取相关参数，使四轴飞行器悬停在原地
    prop_body_ids = robot.find_bodies("m.*_prop")[0]  # 获取螺旋桨的ID
    robot_mass = robot.root_physx_view.get_masses().sum()  # 计算机器人总质量
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()  # 获取重力大小

    # 准备完成
    print("[INFO]: Setup complete...")

    # 定义仿真步进参数
    sim_dt = sim.get_physics_dt()  # 物理仿真时间步长
    sim_time = 0.0  # 仿真时间计数器
    count = 0  # 步数计数器
    # 开始物理仿真
    while simulation_app.is_running():
        # 每2000步重置一次仿真
        if count % 2000 == 0:
            # 重置计数器
            sim_time = 0.0
            count = 0
            # 重置关节状态
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])  # 位置和旋转
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])  # 线速度和角速度
            robot.reset()
            # 输出重置信息
            print(">>>>>>>> Reset!")
        # 对机器人施加作用力（使机器人悬停在原地）
        forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)  # 初始化力矩阵
        torques = torch.zeros_like(forces)  # 初始化扭矩矩阵
        forces[..., 2] = robot_mass * gravity / 4.0  # 在Z轴方向施加力以抵消重力，平均分配到4个螺旋桨

        # 计算每个螺旋桨的RPM值（从力反推）
        k_f = 1e-6  # 推力系数
        force_z = forces[0, :, 2].cpu().numpy()  # 提取Z轴方向的力
        rad_s = np.sqrt(force_z / k_f)  # 根据力计算角速度（弧度/秒）
        rpm = rad_s * 60.0 / (2 * np.pi)  # 转换为RPM
        
        # 使用覆盖式输出，避免命令行太长
        print(f"\r力: {force_z.round(4)}N, RPM: {rpm.round(2)}", end="")


        robot.set_external_force_and_torque(forces, torques, body_ids=prop_body_ids)  # 设置外部力和扭矩
        robot.write_data_to_sim()  # 将数据写入仿真
        # 执行仿真步进
        sim.step()
        # 更新仿真时间
        sim_time += sim_dt
        count += 1
        # 更新机器人状态缓冲区
        robot.update(sim_dt)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用程序
    simulation_app.close()
