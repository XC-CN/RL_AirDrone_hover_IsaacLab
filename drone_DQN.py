import torch
import torch.nn as nn
import numpy as np
import csv
import pandas as pd  # 用于Excel文件处理

import random
import time
import argparse
import sys
import os
from collections import deque
from isaaclab.app import AppLauncher
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib as mpl

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
mpl.rcParams['font.family'] = 'SimHei'  # 设置默认字体为黑体

# 手动实现的SGD步骤
def manual_sgd_step(model, lr):
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param -= lr * param.grad  # 更新参数

def rpm_to_force(rpm, k_f=1e-6):
    """
    将电机转速(RPM)转换为推力
    
    参数:
        rpm: 电机转速，单位为每分钟转数(RPM)
        k_f: 推力系数，默认为1e-6
        
    返回:
        推力值，单位为牛顿(N)
    """
    rad_s = rpm * 2 * 3.14159 / 60.0  # 将RPM转换为弧度/秒
    return k_f * rad_s**2  # 根据平方关系计算推力


def quaternion_to_euler_wxyz(w, x, y, z):
    """
    四元数（w, x, y, z） 转换为 欧拉角（roll, pitch, yaw）
    返回值单位是弧度
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw
def get_euler_angles(self):
    quat = self.robot.data.root_quat_w[0].cpu().numpy()  # shape: (4,)
    w, x, y, z = quat
    return quaternion_to_euler_wxyz(w, x, y, z)

class HoverEnv:
    def __init__(self, device="cpu", headless=True):
        if isinstance(device, torch.device):
            device = str(device)
        self.device = device
        # 转速最大最小值
        self.min_rpm = 2000
        self.max_rpm = 3000

        # 用来追踪悬停时间
        self.hover_time = 0.0
        self.hover_reward_given = False  # 判断是否已经给过奖励
        self.current_step = 0  # 初始化步数计数器

        # 初始化app launcher
        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args_cli = parser.parse_args(args=[])
        args_cli.headless = headless  # 设置无窗口模式
        
        # 添加随机后缀，避免多次运行时路径冲突
        self.sim_id = int(time.time() * 1000) % 10000
        
        self.termination_reason = None  # 用于记录回合结束原因
        
        try:
            self.app_launcher = AppLauncher(args_cli)
            self.simulation_app = self.app_launcher.app

            import isaaclab.sim as sim_utils
            from isaaclab.assets import Articulation
            from isaaclab_assets import CRAZYFLIE_CFG

            sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=device)  # 增加10倍至5ms
            self.sim = sim_utils.SimulationContext(sim_cfg)
            self.sim.set_camera_view(eye=(1.0, 1.0, 1.0), target=(0.0, 0.0, 0.5))

            ground = sim_utils.GroundPlaneCfg()
            ground.func("/World/defaultGroundPlane", ground)

            light = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
            light.func("/World/Light", light)

            # 使用唯一路径名
            drone_path = f"/World/Crazyflie_{self.sim_id}"
            robot_cfg = CRAZYFLIE_CFG.replace(prim_path=drone_path)

            robot_cfg.spawn.func(drone_path, robot_cfg.spawn, translation=robot_cfg.init_state.pos)
            self.robot = Articulation(robot_cfg)

            self.sim.reset()
            self.prop_ids = self.robot.find_bodies("m.*_prop")[0]
            self.dt = self.sim.get_physics_dt()
            print(f"环境初始化成功，无人机路径: {drone_path}")
        except Exception as e:
            print(f"环境初始化失败: {str(e)}")
            raise

    def reset(self):
        self.hover_time = 0.0  # 重置悬停时间
        self.hover_reward_given = False  # 重置奖励状态
        self.current_step = 0  # 重置步数计数器
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)
        self.robot.write_root_pose_to_sim(self.robot.data.default_root_state[:, :7])
        self.robot.write_root_velocity_to_sim(self.robot.data.default_root_state[:, 7:])
        self.robot.reset()
        self.sim.step()
        self.robot.update(self.dt)
        self.stable_hover_steps = 0  # 确保重置稳定悬停计数
        return self._get_state()

    def _get_state(self):
        # 读取位置和速度的每个分量
        x = self.robot.data.root_pos_w[0, 0].item()
        y = self.robot.data.root_pos_w[0, 1].item()
        z = self.robot.data.root_pos_w[0, 2].item()
        vx = self.robot.data.root_vel_w[0, 0].item()
        vy = self.robot.data.root_vel_w[0, 1].item()
        vz = self.robot.data.root_vel_w[0, 2].item()
        quat = self.robot.data.root_quat_w[0].cpu().numpy()[:4]  # w, x, y, z

        # 检查四元数是否合法
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-6:  # 如果四元数范数接近零
            print(f"警告: 检测到接近零范数的四元数 {quat}，使用单位四元数替代")
            quat = np.array([1.0, 0.0, 0.0, 0.0])  # 使用单位四元数
        elif abs(quat_norm - 1.0) > 1e-3:  # 如果四元数未归一化
            print(f"警告: 检测到未归一化的四元数，范数={quat_norm}，进行归一化")
            quat = quat / quat_norm  # 归一化四元数

        # 转换四元数为欧拉角
        try:
            from scipy.spatial.transform import Rotation
            r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # 转换为 x, y, z, w 顺序
            euler = r.as_euler('xyz', degrees=False)  # roll, pitch, yaw
        except Exception as e:
            print(f"警告: 四元数转欧拉角失败: {str(e)}，使用零欧拉角")
            euler = np.array([0.0, 0.0, 0.0])

        # 将所有数据拼接为状态
        state = np.array([x, y, z, vx, vy, vz, euler[0], euler[1], euler[2]], dtype=np.float32)

        return state

    def step(self, rpms):
        # 增加步数计数
        self.current_step += 1
        
        # 限制转速在合理范围内
        rpms = np.clip(rpms, self.min_rpm, self.max_rpm)
        
        # 计算作用于螺旋桨的力和力矩
        forces = torch.zeros(1, 4, 3, device=self.device)
        torques = torch.zeros_like(forces)
        thrusts = []
        for i in range(4):
            thrust = rpm_to_force(rpms[i])
            thrusts.append(thrust)
            forces[0, i, 2] = thrust  # 力作用在z轴方向

        # 施加外力和力矩
        self.robot.set_external_force_and_torque(forces, torques, body_ids=self.prop_ids)
        self.robot.write_data_to_sim()

        # 执行单步物理模拟 - 移除内部迭代
        self.sim.step()
        self.robot.update(self.dt)

        # 获取当前状态
        state = self._get_state()
        
        # 提取状态中的各个分量，使代码更可读
        x, y, z = state[0], state[1], state[2]  # 位置
        vx, vy, vz = state[3], state[4], state[5]  # 速度
        roll, pitch, yaw = state[6], state[7], state[8]  # 姿态角

        # 奖励设置
        # 基于高度的奖励（想要高度接近 0.5）
        height_reward = -5 * abs(z - 0.5)  # 惩罚高度偏离目标

        # 基于俯仰角的奖励（鼓励保持平衡）
        pitch_reward = -abs(pitch)  # 惩罚俯仰角偏离 0.0
        
        # 基于滚动角的奖励（鼓励保持平衡）
        roll_reward = -abs(roll)  # 惩罚滚动角偏离 0.0
        
        # 垂直速度奖励（鼓励垂直速度接近0，实现更稳定的悬停）
        vz_reward = -10 * abs(vz)  # 惩罚垂直速度不为0
        
        # # 悬停奖励：只有在非常接近目标高度0.5时才给予固定奖励
        hover_reward = 0
        
        # 判断是否结束：高度过低或过高，或者俯仰角过大
        done = False

        # 结束条件：高度过低或过高，或者俯仰角过大
        if z < 0.25 or z > 0.75:  # 高度范围
            done = True
            height_reward -= 20  # 高度过大或过小时给负奖励
            if z < 0.1:
                self.termination_reason = "高度过低"
            else:
                self.termination_reason = "高度过高"

        if abs(pitch) > 1:  # 俯仰角超过一定阈值时视为失败
            done = True
            pitch_reward -= 20  # 俯仰角过大时给负奖励
            self.termination_reason = "俯仰角过大"
            
        if abs(roll) > 1:  # 滚动角超过一定阈值时视为失败
            done = True
            roll_reward -= 20  # 滚动角过大时给负奖励
            self.termination_reason = "滚动角过大"
            
        # 稳定悬停结束条件
        # 如果无人机在目标高度附近(0.45-0.55)稳定悬停超过100步，视为成功完成任务
        # 检查对象是否已有stable_hover_steps属性，如果没有则初始化为0
        # 这确保了在第一次调用step方法时能正确初始化稳定悬停计数器
        if not hasattr(self, 'stable_hover_steps'):
            self.stable_hover_steps = 0
            
        # 判断是否在稳定悬停状态
        if (0.49 <= z <= 0.51 and  # 高度非常接近目标
            abs(vz) < 0.05 and     # 垂直速度很小
            abs(roll) < 0.1 and    # Roll角很小
            abs(pitch) < 0.1):     # Pitch角很小
            self.stable_hover_steps += 1
            hover_reward += 100
        else:
            self.stable_hover_steps = 0  # 不满足条件时重置计数
            
        # 如果稳定悬停超过100步，给予高额奖励并结束回合
        if self.stable_hover_steps >= 100:
            hover_reward += 1000.0  # 给予一个很高的奖励
            done = True      # 结束回合
            self.termination_reason = "稳定悬停成功"

        # 综合奖励：包括高度、俯仰角、滚动角、垂直速度和悬停奖励
        reward = height_reward + pitch_reward + roll_reward + vz_reward + hover_reward 

        # 创建包含奖励明细的info字典
        info = {
            "height_reward": height_reward,
            "pitch_reward": pitch_reward,  # 使用实际计算中的系数
            "roll_reward": roll_reward,    # 使用实际计算中的系数
            "vz_reward": vz_reward,
            "hover_reward": hover_reward,
            "total_reward": reward,
            "termination_reason": self.termination_reason if done else None
        }

        # 返回新的状态、奖励、是否结束以及奖励明细信息
        return state, reward, done, info


    def sample_action(self):
        """
        随机采样动作，采样sqrt(RPM)并转换为RPM值
        """
        # 在平方根空间中均匀采样
        sqrt_rpms = np.random.uniform(
            np.sqrt(self.min_rpm), 
            np.sqrt(self.max_rpm), 
            size=(4,)
        )
        # 转换回RPM
        return sqrt_rpms ** 2

    def close(self):
        """安全关闭仿真环境"""
        try:
            if hasattr(self, 'simulation_app') and self.simulation_app is not None:
                self.simulation_app.close()
                print(f"环境 ID {self.sim_id} 已关闭")
        except Exception as e:
            print(f"关闭环境时出错: {str(e)}")
            # 尝试强制退出
            import gc
            gc.collect()  # 强制垃圾回收


class DQN(nn.Module):
    """
    深度Q网络(DQN)模型定义
    
    用于无人机控制的神经网络模型，接收状态输入并输出每个动作的Q值
    对应RPM的平方根，而不是直接输出RPM，使动作空间与物理效果更接近线性关系
    
    参数:
        state_dim: 状态空间维度，默认为9，包含位置、速度、姿态等信息
        action_dim: 动作空间维度，默认为4，对应四个电机的RPM值
    """
    def __init__(self, state_dim=9, action_dim=4):
        super().__init__()
        # 定义一个三层全连接神经网络
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),  # 输入层到第一隐藏层，64个神经元
            nn.ReLU(),                 # 激活函数
            nn.Linear(64, 64),         # 第一隐藏层到第二隐藏层，64个神经元
            nn.ReLU(),                 # 激活函数
            nn.Linear(64, action_dim)  # 第二隐藏层到输出层，输出每个动作的Q值
        )
        # RPM范围的平方根
        self.min_sqrt_rpm = np.sqrt(4900)  # sqrt(min_rpm)
        self.max_sqrt_rpm = np.sqrt(7000)  # sqrt(max_rpm)

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入状态张量
            
        返回:
            每个动作的Q值，对应RPM的平方根值
        """
        # 网络输出RPM的平方根值
        sqrt_rpm = self.net(x)
        
        # 限制输出在合理的sqrt(RPM)范围内
        sqrt_rpm = torch.clamp(sqrt_rpm, self.min_sqrt_rpm, self.max_sqrt_rpm)
        
        return sqrt_rpm


def train_drone(total_episodes=3000, headless=True, model_save_path="models/dqn_drone/model"):
    """训练无人机使用DQN算法
    
    参数:
        total_episodes: 总训练回合数
        headless: 是否使用无窗口模式
        model_save_path: 模型保存路径
    """
    # 创建模型保存目录
    model_dir = os.path.dirname(model_save_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"创建模型保存目录: {model_dir}")
    
    # 创建环境
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = HoverEnv(device=device, headless=headless)
    
    # 初始化网络
    q_net = DQN(9, 4).to(device)
    target_net = DQN(9, 4).to(device)
    target_net.load_state_dict(q_net.state_dict())

    # 设置训练参数
    buffer = deque(maxlen=10000)
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0 
    epsilon_decay = 0.995
    epsilon_min = 0.01
    lr = 1e-3  # 学习率
    target_update_freq = 10  # 目标网络更新频率
    
    # 初始化训练统计
    start_time = time.time()
    all_rewards = []
    all_lengths = []
    update_count = 0
    
    termination_reasons = []  # 用于统计各种结束原因的列表
    
    print(f"开始训练DQN算法，总回合数: {total_episodes}")
    print(f"设备: {device}, 隐藏窗口: {headless}")
    print(f"模型将保存到: {model_save_path}")
    
    try:
        for episode in range(total_episodes):
            state = env.reset()
            total_reward = 0
            episode_length = 0
            
            for step in range(300):  # 每回合最多300步
                episode_length += 1
                
                # 选择动作
                state_tensor = torch.tensor(state, device=device).float()
                with torch.no_grad():
                    sqrt_action = q_net(state_tensor).cpu().numpy()
                    # 将sqrt(RPM)转换为RPM
                    action = sqrt_action ** 2

                # epsilon-贪婪策略
                if random.random() < epsilon:
                    action = env.sample_action()  # 这里sample_action已经返回实际RPM

                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 存储经验 - 注意：存储的是sqrt(RPM)
                buffer.append((state, sqrt_action, reward, next_state, done))
                
                # 更新状态和累积奖励
                state = next_state
                total_reward += reward

                # 执行训练
                if len(buffer) >= batch_size:
                    batch = random.sample(buffer, batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    # 先将列表转换为numpy数组，再创建tensor，提高性能
                    states = torch.tensor(np.array(states), device=device).float()
                    actions = torch.tensor(np.array(actions), device=device).float()
                    rewards = torch.tensor(np.array(rewards), device=device).float()
                    next_states = torch.tensor(np.array(next_states), device=device).float()
                    dones = torch.tensor(np.array(dones), device=device).float()

                    # 计算当前Q值
                    q_values = q_net(states)
                    q_selected = q_values
                    
                    # 计算目标Q值
                    next_q = target_net(next_states)
                    target = rewards.unsqueeze(1) + gamma * next_q.mean(1, keepdim=True) * (1 - dones.unsqueeze(1))

                    # 计算损失并更新网络
                    loss = nn.MSELoss()(q_selected.mean(1, keepdim=True), target.detach())
                    q_net.zero_grad()  # 清除梯度
                    loss.backward()    # 反向传播
                    manual_sgd_step(q_net, lr)
                    update_count += 1

                # 如果回合结束，记录统计信息并退出循环
                if done:                 
                    break

            # 收集统计信息
            all_rewards.append(total_reward)
            all_lengths.append(episode_length)
            
            # 更新epsilon值
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            
            # 定期更新目标网络
            if episode % target_update_freq == 0:
                target_net.load_state_dict(q_net.state_dict())
            
            # 计算平均奖励和步长
            avg_reward = sum(all_rewards[-100:]) / min(len(all_rewards), 100)
            avg_length = sum(all_lengths[-100:]) / min(len(all_lengths), 100)
            
            # 计算训练用时
            elapsed_time = int(time.time() - start_time)
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # 使用普通输出方式显示进度
            print(
                f"回合: {episode+1:4d}/{total_episodes:4d} | "
                f"累计奖励: {total_reward:8.1f} | "
                f"每步平均奖励: {avg_reward:8.1f} | "
                f"步数: {episode_length:4d} | "
                f"ε: {epsilon:.3f} | "
                # f"更新次数: {update_count:6d} | "
                f"用时: {hours:02d}:{minutes:02d}:{seconds:02d} | "
                f"结束原因: {info.get('termination_reason', '未知原因') if done else '进行中'}"
            )
            
            # # 在回合结束时记录原因
            # if done:
            #     reason = info.get("termination_reason", "未知原因")
            #     termination_reasons.append(reason)
            
    except Exception as e:
        import traceback
        print(f"训练过程发生错误: {str(e)}")
        print(traceback.format_exc())  # 打印完整堆栈跟踪
    
    finally:
        # 保存最终模型
        final_model_path = f"{model_save_path}_final.pth"
        torch.save(q_net, final_model_path)
        print(f"\n模型已保存到 {final_model_path}")
        
        # 关闭环境
        env.close()
        
        # 打印训练总结
        print("\n===== 训练总结 =====")
        print(f"总回合数: {len(all_rewards)}")
        print(f"最终探索率(epsilon): {epsilon:.4f}")
        print(f"网络更新次数: {update_count}")
        if all_rewards:
            print(f"最后100回合平均奖励: {sum(all_rewards[-100:]) / min(len(all_rewards), 100):.2f}")
            print(f"最好单回合奖励: {max(all_rewards):.2f}")
        
        return q_net


def test_drone(model_path="models/dqn_drone/model_final.pth", headless=False, test_episodes=1, use_chinese=True):
    """测试训练好的DQN模型，并绘制高度随时间变化的图表
    
    参数:
        model_path: 模型文件路径
        headless: 是否使用无窗口模式
        test_episodes: 测试回合数，默认为1回合
        use_chinese: 是否在图表中使用中文，如果系统不支持中文字体可设为False
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 根据参数选择字体设置
    if use_chinese:
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            title_height = '高度随时间变化'
            title_horizontal = '水平偏移随时间变化'
            title_velocity = '垂直速度随时间变化'
            title_tilt = '倾角随时间变化'
            title_motors = '电机RPM随时间变化'
            label_target = '目标高度'
            label_lower = '下限'
            label_upper = '上限'
            motor_labels = [f'电机 {i+1}' for i in range(4)]
        except:
            print("中文字体加载失败，自动切换为英文显示")
            use_chinese = False
    
    if not use_chinese:
        title_height = 'Height vs Time'
        title_horizontal = 'Horizontal Offset vs Time'
        title_velocity = 'Vertical Velocity vs Time'
        title_tilt = 'Tilt Angle vs Time'
        title_motors = 'Motor RPM vs Time'
        label_target = 'Target Height'
        label_lower = 'Lower Limit'
        label_upper = 'Upper Limit'
        motor_labels = [f'Motor {i+1}' for i in range(4)]
    
    try:
        # 先检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"错误：模型文件 {model_path} 不存在!")
            return
            
        # 先尝试加载模型，防止环境创建后模型加载失败导致资源不释放
        try:
            print(f"尝试加载模型: {model_path}...")
            q_net = torch.load(model_path, map_location=device)
            print(f"模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return
        
        # 只有在headless模式(无窗口)下才需要创建figures文件夹用于保存图表
        if headless:
            # 创建figures文件夹，用于保存图表
            if not os.path.exists('figures'):
                os.makedirs('figures')
                print("创建figures文件夹用于保存图表")
        
        # 创建data文件夹，用于保存Excel数据
        if not os.path.exists('data'):
            os.makedirs('data')
            print("创建data文件夹用于保存测试数据")
            
        # 使用简洁的文件名 - 只使用模型名，不带时间戳
        model_name = os.path.basename(model_path).replace('.pth', '')
        excel_filename = f"data/drone_test_{model_name}"
        
        # 初始化数据框用于收集所有测试数据
        columns = [
            "回合", "步数", "X位置", "Y位置", "Z位置", 
            "X速度", "Y速度", "Z速度", "Roll", "Pitch", "Yaw",
            "水平偏移", "电机1_RPM", "电机2_RPM", "电机3_RPM", "电机4_RPM",
            "高度奖励", "倾角奖励", "滚动角奖励", "垂直速度奖励", "悬停奖励", 
            "当前步奖励", "累积奖励"
        ]
        all_data = []  # 用于收集所有数据行
        
        print(f"数据将保存到: {excel_filename}.xlsx")
        
        # 创建环境（只有模型成功加载后再创建环境）
        print("创建仿真环境...")
        env = HoverEnv(device=device, headless=headless)
        
        # 测试统计数据
        test_count = 0
        total_reward_sum = 0
        max_reward = float('-inf')
        min_reward = float('inf')
        max_steps = 0
        total_steps = 0
        
        if test_episodes > 0:
            print(f"\n开始测试，共{test_episodes}个回合...")
        else:
            print("\n开始无限循环测试，按Ctrl+C停止...")
        
        # 测试循环
        while test_episodes == 0 or test_count < test_episodes:
            test_count += 1
            print(f"\n开始测试回合 {test_count}")
            
            try:
                state = env.reset()
                total_reward = 0
                done = False
                step = 0
                
                # 用于记录高度数据的列表
                heights = []
                time_steps = []
                rewards = []
                horizontal_offsets = []
                velocities_z = []
                tilt_angles = []
                actions_data = []
                
                while not done and step < 500:  # 测试时允许更长的回合
                    step += 1
                    
                    # 使用模型选择动作，不使用探索
                    state_tensor = torch.tensor(state, device=device).float()
                    with torch.no_grad():
                        sqrt_action = q_net(state_tensor).cpu().numpy()
                        # 将sqrt(RPM)转换为RPM
                        action = sqrt_action ** 2
                    
                    # 执行动作
                    next_state, reward, done, info = env.step(action)
                    total_reward += reward
                    
                    # 记录数据
                    heights.append(state[2])  # 高度 Z
                    time_steps.append(step)
                    rewards.append(reward)
                    horizontal_offsets.append(np.sqrt(state[0]**2 + state[1]**2))  # 水平偏移
                    velocities_z.append(state[5])  # 垂直速度
                    tilt_angles.append(np.sqrt(state[6]**2 + state[7]**2))  # 倾角
                    actions_data.append(action)  # 记录动作
                    
                    # 从info字典中获取奖励明细
                    height_reward = info["height_reward"]
                    pitch_reward_factor = info["pitch_reward"]
                    roll_reward_factor = info["roll_reward"]
                    vz_reward = info["vz_reward"]
                    hover_reward = info["hover_reward"]
                    
                    # 水平位置偏移
                    horizontal_offset = np.sqrt(state[0]**2 + state[1]**2)
                    
                    # 如果是第一步，显示标题，让用户知道正在测试中
                    if step == 1:
                        print(f"测试中...(详细数据记录到 {excel_filename}.xlsx)")
                    
                    # 只显示简短的状态信息
                    if step % 50 == 0:  # 每50步打印一次简短状态
                        print(f"回合 {test_count} 步 {step} | 高度: {state[2]:.3f}m | 奖励: {reward:.2f} | 总奖励: {total_reward:.2f}")
                    
                    # 将数据写入Excel文件 - 为了记录和分析，同时保存sqrt(RPM)和RPM
                    all_data.append([
                        test_count, step, 
                        state[0], state[1], state[2],  # 位置
                        state[3], state[4], state[5],  # 速度
                        state[6], state[7], state[8],  # 欧拉角
                        horizontal_offset,             # 水平偏移
                        action[0], action[1], action[2], action[3],  # 实际RPM
                        height_reward, pitch_reward_factor, roll_reward_factor, vz_reward, hover_reward,  # 奖励明细
                        reward, total_reward           # 总奖励
                    ])
                    
                    # 更新状态
                    state = next_state
                
                # 更新统计数据
                total_reward_sum += total_reward
                avg_reward = total_reward_sum / test_count
                max_reward = max(max_reward, total_reward)
                min_reward = min(min_reward, total_reward)
                max_steps = max(max_steps, step)
                total_steps += step
                avg_steps = total_steps / test_count
                
                # 输出回合结果
                result_status = "成功" if step >= 400 else "失败"
                print(f"\n回合 {test_count} {result_status} | 总奖励: {total_reward:.1f} | 持续步数: {step}")
                
                # 保存Excel数据
                # 在任何模式下都只保存第一回合数据
                save_excel = test_count == 1
                
                if save_excel and 'all_data' in locals() and all_data:
                    try:
                        # 创建DataFrame并保存为Excel
                        df = pd.DataFrame(all_data, columns=columns)
                        # 明确标记为"第一回合"数据
                        excel_filename = f"{excel_filename}"
                        df.to_excel(f"{excel_filename}.xlsx", index=False)
                        print(f"第一回合数据已保存到: {excel_filename}.xlsx")
                        
                        # 保存后清空all_data以避免累积后续回合的数据
                        print("已保存第一回合数据，后续回合将不再保存表格数据")
                        all_data = []  # 清空数据集合
                    except Exception as e:
                        print(f"保存Excel文件时出错: {str(e)}")
                
                # 保存图表
                # 使用固定文件名保存图表，避免生成过多文件
                model_name = os.path.basename(model_path).replace('.pth', '')
                
                # 创建一个2x2的子图布局
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # 1. 高度随时间变化图
                axes[0, 0].plot(time_steps, heights, 'b-', linewidth=2)
                axes[0, 0].axhline(y=0.5, color='r', linestyle='--', label=label_target)
                axes[0, 0].axhline(y=0.4, color='g', linestyle=':', label=label_lower)
                axes[0, 0].axhline(y=0.6, color='g', linestyle=':', label=label_upper)
                axes[0, 0].set_title(title_height)
                axes[0, 0].set_xlabel('步数')
                axes[0, 0].set_ylabel('高度 (m)')
                axes[0, 0].grid(True)
                axes[0, 0].legend()
                
                # 2. 水平偏移随时间变化图
                axes[0, 1].plot(time_steps, horizontal_offsets, 'g-', linewidth=2)
                axes[0, 1].set_title(title_horizontal)
                axes[0, 1].set_xlabel('步数')
                axes[0, 1].set_ylabel('水平偏移 (m)')
                axes[0, 1].grid(True)
                
                # 3. 垂直速度随时间变化图
                axes[1, 0].plot(time_steps, velocities_z, 'm-', linewidth=2)
                axes[1, 0].set_title(title_velocity)
                axes[1, 0].set_xlabel('步数')
                axes[1, 0].set_ylabel('垂直速度 (m/s)')
                axes[1, 0].grid(True)
                
                # 4. 倾角随时间变化图
                axes[1, 1].plot(time_steps, tilt_angles, 'c-', linewidth=2)
                axes[1, 1].set_title(title_tilt)
                axes[1, 1].set_xlabel('步数')
                axes[1, 1].set_ylabel('倾角 (rad)')
                axes[1, 1].grid(True)
                
                # 保存主图表
                plt.tight_layout()
                
                # 确保figures文件夹存在
                if not os.path.exists('figures'):
                    os.makedirs('figures')

                
                fig_path = f'figures/drone_test_{model_name}.png'
                    
                plt.savefig(fig_path, dpi=300)  # 提高分辨率
                print(f"图表已保存到: {fig_path}")
                plt.close()
                
                # 创建动作值图表
                fig, ax = plt.subplots(figsize=(12, 6))
                actions_data = np.array(actions_data)
                for i in range(4):
                    ax.plot(time_steps, actions_data[:, i], label=motor_labels[i])
                ax.set_title(title_motors)
                ax.set_xlabel('步数')
                ax.set_ylabel('RPM')
                ax.grid(True)
                ax.legend()
                plt.tight_layout()
                
                # 文件命名策略简化
                action_fig_path = f'figures/drone_actions_{model_name}.png'
                    
                plt.savefig(action_fig_path, dpi=300)  # 提高分辨率
                print(f"动作图表已保存到: {action_fig_path}")
                plt.close()
                
                # 如果结束了一个回合且测试次数已达到要求，则跳出循环
                if test_episodes > 0 and test_count >= test_episodes:
                    break
                    
            except Exception as e:
                print(f"\n回合 {test_count} 发生错误: {str(e)}")
            
            # 检查是否需要结束测试
            if test_episodes > 0 and test_count >= test_episodes:
                print(f"\n已完成 {test_episodes} 个测试回合")
                break

    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n测试发生严重错误: {str(e)}")
    
    finally:
        # 确保环境关闭
        if 'env' in locals():
            try:
                print("正在关闭环境...")
                env.close()
                print("环境已关闭")
            except Exception as e:
                print(f"关闭环境时发生错误: {str(e)}")
        
        # 打印测试总结
        if 'test_count' in locals() and test_count > 0:
            print("\n===== 测试统计 =====")
            print(f"总测试次数: {test_count}")
            if 'avg_reward' in locals():
                print(f"平均奖励: {avg_reward:.2f}")
                print(f"最高奖励: {max_reward:.2f}")
                print(f"最低奖励: {min_reward:.2f}")
            if 'avg_steps' in locals():
                print(f"平均步数: {avg_steps:.1f}")
                print(f"最大步数: {max_steps}")
            print("====================")
            
            # 确保在任何模式下都不再在程序结束时保存数据
            if 'all_data' in locals() and all_data:
                print("\n注意: 程序仅保存第一回合数据，测试结束时不再保存额外数据。")
                # 确保清空all_data，防止累积
                all_data = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN算法训练无人机悬停")
    parser.add_argument("--test", action="store_true", help="测试模式而非训练模式")
    parser.add_argument("--model", type=str, default="models/dqn_drone/model_final.pth", help="测试时使用的模型路径")
    parser.add_argument("--episodes", type=int, default=500, help="训练总回合数")
    parser.add_argument("--window", action="store_true", help="显示模拟窗口，此模式下图表和Excel数据会覆盖保存，且自动进入无限循环模式(除非指定--loop)")
    parser.add_argument("--loop", type=int, default=1, help="测试回合数，默认为1，设为0表示无限循环")
    parser.add_argument("--english", action="store_true", help="图表使用英文显示，解决中文显示乱码问题")
    
    args = parser.parse_args()
    
    # 当使用--window参数时，如果没有明确指定loop，则设置为无限循环
    if args.window and not any('--loop' in arg for arg in sys.argv):
        print("检测到窗口模式，自动设置为无限循环测试。按Ctrl+C可随时停止。")
        args.loop = 0
    
    if args.test:
        test_drone(
            model_path=args.model, 
            headless=not args.window, 
            test_episodes=args.loop,
            use_chinese=not args.english
        )
    else:
        train_drone(
            total_episodes=args.episodes, 
            headless=not args.window, 
            model_save_path="models/dqn_drone/model"
        )
