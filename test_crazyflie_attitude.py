import torch
import numpy as np
from isaaclab.app import AppLauncher
import argparse
import time

def main():
    """探索Crazyflie无人机姿态数据的存储格式"""
    
    # 初始化Isaac模拟器
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args(args=[])
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # 导入必要的模块
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab_assets import CRAZYFLIE_CFG

    # 设置模拟环境
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.5])

    # 创建地面
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/defaultGroundPlane", ground)

    # 添加光源
    light = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
    light.func("/World/Light", light)

    # 配置无人机模型
    robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    robot = Articulation(robot_cfg)

    # 重置模拟环境
    sim.reset()
    dt = sim.get_physics_dt()
    robot.update(dt)
    
    # 找到螺旋桨的ID以施加力
    prop_ids = robot.find_bodies("m.*_prop")[0]

    # 检查并打印机器人数据结构
    print("\n===== 检查机器人数据结构 =====")
    for attr_name in dir(robot.data):
        if not attr_name.startswith('_'):
            try:
                attr = getattr(robot.data, attr_name)
                if isinstance(attr, (torch.Tensor, np.ndarray)):
                    print(f"robot.data.{attr_name}: shape={attr.shape}, type={type(attr)}")
                else:
                    print(f"robot.data.{attr_name}: type={type(attr)}")
            except Exception as e:
                print(f"无法获取 robot.data.{attr_name}: {e}")
    
    # 特别检查root_pos_w
    print("\n===== 检查root_pos_w的内容 =====")
    if hasattr(robot.data, 'root_pos_w'):
        print(f"root_pos_w shape: {robot.data.root_pos_w.shape}")
        print(f"root_pos_w content: {robot.data.root_pos_w}")
    
    # 检查是否有专门的四元数或姿态变量
    print("\n===== 查找姿态相关变量 =====")
    attitude_keywords = ['quat', 'orient', 'rot', 'angle', 'att', 'pose']
    for attr_name in dir(robot.data):
        if any(keyword in attr_name.lower() for keyword in attitude_keywords):
            try:
                attr = getattr(robot.data, attr_name)
                print(f"发现可能的姿态相关变量: robot.data.{attr_name}")
                if isinstance(attr, (torch.Tensor, np.ndarray)):
                    print(f"  形状: {attr.shape}")
                    print(f"  内容: {attr}")
            except Exception as e:
                print(f"无法获取 robot.data.{attr_name}: {e}")
    
    # 尝试施加力矩使无人机倾斜，然后观察姿态变化
    print("\n===== 施加力矩观察姿态变化 =====")
    print("初始状态:")
    print_drone_state(robot)
    
    # 施加不平衡的力，使无人机倾斜
    forces = torch.zeros(1, 4, 3, device=device)
    torques = torch.zeros_like(forces)
    
    # 只给右侧两个螺旋桨施加力，使无人机倾斜
    forces[0, 0, 2] = 0.1  # 右前
    forces[0, 3, 2] = 0.1  # 右后
    
    robot.set_external_force_and_torque(forces, torques, body_ids=prop_ids)
    robot.write_data_to_sim()
    
    # 模拟一段时间并观察姿态变化
    print("\n模拟中...")
    for i in range(50):  # 模拟50个时间步
        sim.step()
        robot.update(dt)
        if i % 10 == 0:  # 每10步打印一次
            print(f"\n时间步 {i}:")
            print_drone_state(robot)
    
    # 清理并关闭模拟器
    simulation_app.close()
    print("\n模拟结束")

def print_drone_state(robot):
    """打印无人机的当前状态，重点关注姿态信息"""
    print(f"位置: {robot.data.root_pos_w[0, :3].cpu().numpy()}")
    
    # 尝试获取并打印四元数
    if robot.data.root_pos_w.shape[1] > 3:
        try:
            quat_data = robot.data.root_pos_w[0, 3:7].cpu().numpy()
            print(f"四元数 (如果存在): {quat_data}")
        except:
            print("无法获取四元数数据")
    
    # 打印速度和角速度
    print(f"线速度: {robot.data.root_vel_w[0, :3].cpu().numpy()}")
    if robot.data.root_vel_w.shape[1] > 3:
        print(f"角速度: {robot.data.root_vel_w[0, 3:].cpu().numpy()}")
    
    # 检查其他可能包含姿态信息的属性
    attitude_attrs = ['root_quat_w', 'root_rot_w', 'root_orient_w']
    for attr in attitude_attrs:
        if hasattr(robot.data, attr):
            print(f"{attr}: {getattr(robot.data, attr)[0].cpu().numpy()}")

if __name__ == "__main__":
    main() 