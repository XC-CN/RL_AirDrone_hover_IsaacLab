import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import argparse
from isaaclab.app import AppLauncher
import time
from torch.distributions import Normal
import os

# 全局变量
total_steps = 0

def rpm_to_force(rpm, k_f=1e-6):
    rad_s = rpm * 2 * 3.14159 / 60.0
    return k_f * rad_s**2

class HoverEnv:
    def __init__(self, device="cpu", enable_gui=True, enable_wind=False):
        self.device = device

        # 转速设置
        self.min_rpm = 4500
        self.max_rpm = 5500
        self.enable_gui = enable_gui  # 是否启用GUI
        
        # 风力参数
        self.enable_wind = enable_wind  # 是否启用风力
        self.wind_strength = 0.0  # 初始风力强度为0，将在更新风力时设置
        self.wind_direction = 0.0  # 初始风向，将在更新风力时设置
        self.wind_direction_deg = 0.0  # 风向(角度)，用于输出显示
        
        # 风力随机范围
        self.min_wind_strength = 0.2  # 最小风力强度 (N)
        self.max_wind_strength = 0.4  # 最大风力强度 (N)
        
        # 风力变化
        self.wind_variation_amplitude = 0.2  # 风力变化幅度 (相对于基础风力的比例)
        self.wind_update_freq = 0.2  # 风力更新频率 (Hz)
        self.wind_update_counter = 0
        self.wind_update_steps = int(1.0 / (self.wind_update_freq * 0.01))  # 假设dt=0.01s
        
        # 如果启用风力，生成初始风力和风向
        if self.enable_wind:
            self._reset_wind()
        
        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args_cli = parser.parse_args(args=[])
        
        # 根据GUI设置修改参数
        if not self.enable_gui:
            args_cli.headless = True  # 无头模式，不显示GUI
        
        self.app_launcher = AppLauncher(args_cli)
        self.simulation_app = self.app_launcher.app

        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab_assets import CRAZYFLIE_CFG

        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        
        # 仅在GUI模式下设置摄像机
        if self.enable_gui:
            self.sim.set_camera_view(eye=(1.0, 1.0, 1.0), target=(0.0, 0.0, 0.5))

        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/defaultGroundPlane", ground)

        # 仅在GUI模式下添加光源
        if self.enable_gui:
            light = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
            light.func("/World/Light", light)

        # 创建机器人配置
        # 使用兼容的方法设置无人机配置
        robot_cfg = CRAZYFLIE_CFG  # 直接使用原始配置
        robot_cfg.prim_path = "/World/Crazyflie"  # 直接设置属性
        
        # 使用spawn函数创建无人机
        robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
        self.robot = Articulation(robot_cfg)

        self.sim.reset()
        self.prop_ids = self.robot.find_bodies("m.*_prop")[0]
        self.dt = self.sim.get_physics_dt()
        
        # 目标位置
        self.target_position = np.array([0.0, 0.0, 1.0])  # 目标位置(x, y, z)=(0, 0, 1)
        self.target_height = 1.0
        
        # 终止条件阈值
        self.min_height = 0.2
        self.max_height = 1.5
        
        # XY平面边界
        self.xy_boundary = 1.0  # XY平面最大允许偏离原点±1米

    def _reset_wind(self):
        """重置风力，随机生成新的风力强度和风向"""
        # 随机生成风力强度（范围在min_wind_strength到max_wind_strength之间）
        self.wind_strength = np.random.uniform(self.min_wind_strength, self.max_wind_strength)
        
        # 随机生成风向（范围在-π到π之间，即所有可能的方向）
        self.wind_direction = np.random.uniform(-np.pi, np.pi)
        self.wind_direction_deg = np.degrees(self.wind_direction)  # 存储角度值，用于显示
        
        # 计算风力分量
        self.wind_force_x = self.wind_strength * np.cos(self.wind_direction)
        self.wind_force_y = self.wind_strength * np.sin(self.wind_direction)
        self.wind_force_z = 0.0  # 假设风只在水平面吹
        
        # 打印风力信息
        if self.enable_gui:
            print(f"新的风力设置: 强度={self.wind_strength:.2f}N, 风向={self.wind_direction_deg:.1f}°")

    def _update_wind_force(self):
        """更新风力，增加随机变化"""
        # 风力强度随机变化
        variation = 1.0 + np.random.uniform(-self.wind_variation_amplitude, self.wind_variation_amplitude)
        current_strength = self.wind_strength * variation
        
        # 风向也可以有微小变化
        direction_variation = np.random.uniform(-0.1, 0.1)  # ±0.1弧度的变化
        current_direction = self.wind_direction + direction_variation
        
        # 计算风力分量
        self.wind_force_x = current_strength * np.cos(current_direction)
        self.wind_force_y = current_strength * np.sin(current_direction)
        
        if self.enable_gui and self.enable_wind:
            # 如果启用了GUI且风力不为零，打印风力信息
            wind_dir_deg = np.degrees(current_direction)
            print(f"\r当前风力: {current_strength:.2f}N, 风向: {wind_dir_deg:.1f}°", end="")

    def reset(self):
        # 如果启用了风力，在每个回合开始时重新随机生成风力和风向
        if self.enable_wind:
            self._reset_wind()
        
        # 设置初始姿态，默认为水平姿态
        roll_init = 0.0  # Roll初始值为0（水平姿态）
        pitch_init = 0.0 # Pitch初始值为0
        yaw_init = 0.0   # Yaw可以保持为0
        
        # 随机化初始高度，在目标高度附近，增加训练多样性
        initial_height = self.target_position[2] + np.random.uniform(-0.1, 0.1)
        
        # 随机化初始XY位置，小范围内，帮助学习XY稳定控制
        initial_x = np.random.uniform(-0.1, 0.1)
        initial_y = np.random.uniform(-0.1, 0.1)
        
        # 欧拉角转四元数 (使用ZYX顺序，符合大多数物理引擎)
        cy = np.cos(yaw_init * 0.5)
        sy = np.sin(yaw_init * 0.5)
        cp = np.cos(pitch_init * 0.5)
        sp = np.sin(pitch_init * 0.5)
        cr = np.cos(roll_init * 0.5)
        sr = np.sin(roll_init * 0.5)
        
        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = sy * cp * sr + cy * sp * cr
        qz = sy * cp * cr - cy * sp * sr
        
        # 创建四元数数组
        initial_quat = torch.tensor([qx, qy, qz, qw], device=self.device).unsqueeze(0)
        
        # 设置默认关节状态
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)
        
        # 设置初始位置，保持默认位置，但使用我们计算的四元数和位置
        root_state = self.robot.data.default_root_state.clone()
        
        # 设置位置
        root_state[:, 0] = initial_x  # X坐标
        root_state[:, 1] = initial_y  # Y坐标
        root_state[:, 2] = initial_height  # Z坐标
        
        # 设置姿态（四元数）
        root_state[:, 3:7] = initial_quat  # 替换四元数部分
        
        # 设置初始速度为零，减少初始不稳定性
        root_state[:, 7:13] = 0.0
        
        # 写入位置和四元数
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(root_state[:, 7:])
        
        self.robot.reset()
        self.sim.step()
        self.robot.update(self.dt)
        
        # 获取初始状态
        state = self._get_state()
        
        return state

    def _get_state(self):
        # 获取位置 (x, y, z)
        pos = self.robot.data.root_pos_w[0, :3].cpu().numpy()
        
        # 获取姿态 (roll, pitch, yaw)，从四元数转换
        quat = self.robot.data.root_quat_w[0].cpu().numpy()
        roll, pitch, yaw = self._quat_to_euler(quat)
        
        # 获取线速度
        lin_vel = self.robot.data.root_vel_w[0, :3].cpu().numpy()
        
        # 获取角速度
        ang_vel = self.robot.data.root_ang_vel_w[0, :3].cpu().numpy()
        
        # 计算与目标位置的距离误差
        position_error = pos - self.target_position
        
        # 返回状态空间：
        # [当前位置(3), 目标位置(3), 姿态角(3), 线速度(3), 角速度(3), 位置误差(3)]
        return np.concatenate([
            pos,                  # 当前位置 [x, y, z]
            self.target_position, # 目标位置 [target_x, target_y, target_z]
            [roll, pitch, yaw],   # 姿态角 [roll, pitch, yaw]
            lin_vel,              # 线速度 [vx, vy, vz]
            ang_vel,              # 角速度 [wx, wy, wz]
            position_error        # 位置误差 [x_error, y_error, z_error]
        ], dtype=np.float32)

    def _quat_to_euler(self, quat):
        """四元数转欧拉角 (roll, pitch, yaw)"""
        x, y, z, w = quat
        
        # Roll (绕X轴)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (绕Y轴)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # 使用90度，如果越界
        else:
            pitch = np.arcsin(sinp)
            
        # Yaw (绕Z轴)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    def step(self, rpms):
        rpms = np.clip(rpms, self.min_rpm, self.max_rpm)
        forces = torch.zeros(1, 4, 3, device=self.device)
        torques = torch.zeros_like(forces)
        for i in range(4):
            thrust = rpm_to_force(rpms[i])
            forces[0, i, 2] = thrust

        # 添加到转子的力和扭矩
        self.robot.set_external_force_and_torque(forces, torques, body_ids=self.prop_ids)
        
        # 如果启用了风力，添加风力影响
        if self.enable_wind:
            # 更新风力计数器
            self.wind_update_counter += 1
            if self.wind_update_counter >= self.wind_update_steps:
                self._update_wind_force()
                self.wind_update_counter = 0
                
            # 在机身上施加风力 (body_ids=None表示施加到整个无人机上)
            wind_force = torch.tensor([[self.wind_force_x, self.wind_force_y, self.wind_force_z]], 
                                    dtype=torch.float32, device=self.device).unsqueeze(1)
            wind_torque = torch.zeros(1, 1, 3, dtype=torch.float32, device=self.device)  # 风不直接产生扭矩
            
            # 找到机身ID (避开螺旋桨)
            body_ids = self.robot.find_bodies("body")[0]  # 使用"body"而不是"frame"
            self.robot.set_external_force_and_torque(wind_force, wind_torque, body_ids=body_ids)
        
        self.robot.write_data_to_sim()

        for _ in range(4):
            self.sim.step()
            self.robot.update(self.dt)

        state = self._get_state()
        reward = self._calculate_reward(state)
        done = self._check_done(state)
        
        return state, reward, done, {}

    def _calculate_reward(self, state):
        # 提取状态信息（从新的状态空间中正确提取）
        # 状态空间：[当前位置(3), 目标位置(3), 姿态角(3), 线速度(3), 角速度(3), 位置误差(3)]
        x_error, y_error, z_error = state[15], state[16], state[17]  # 位置误差
        
        # 计算到目标位置的欧氏距离
        position_error = np.array([x_error, y_error, z_error])
        distance_to_target = np.linalg.norm(position_error)
        
        # 奖励函数简化：仅基于到目标位置的距离计算
        # 距离越小，奖励越大
        reward = 10.0 * np.exp(-10.0 * distance_to_target**2)
        
        return reward

    def _check_done(self, state):
        # 获取位置信息
        x, y, z = state[0], state[1], state[2]  # x, y, z 坐标
        
        # 判断高度是否超出范围
        height_out_of_bounds = z < self.min_height or z > self.max_height
        
        # 判断XY平面位置是否超出范围
        xy_out_of_bounds = abs(x) > self.xy_boundary or abs(y) > self.xy_boundary
        
        # 任一条件满足即终止
        return height_out_of_bounds or xy_out_of_bounds

    def sample_action(self):
        """随机采样动作，生成连续的转速值"""
        # 生成连续的随机转速值
        return np.random.uniform(self.min_rpm, self.max_rpm, size=(4,))

    def close(self):
        self.simulation_app.close()

class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # 平均值输出层
        self.mean_layer = nn.Linear(256, action_dim)
        
        # 初始标准差，作为可学习参数，较小的初始值提高稳定性
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)
        
        # 动作范围限制
        self.action_scale = 1.0
        self.action_bias = 0.0
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0.0)
        
        # 特殊初始化平均值层，输出接近零
        nn.init.orthogonal_(self.mean_layer.weight, gain=0)
        nn.init.constant_(self.mean_layer.bias, 0.0)
    
    def forward(self, state):
        x = self.network(state)
        
        # 动作均值，使用tanh确保在[-1,1]范围内
        mean = torch.tanh(self.mean_layer(x))
        
        # 标准差，限制在合理范围内
        log_std = self.log_std.clamp(-20, 2)  # 避免过大或过小的标准差
        std = torch.exp(log_std)
        
        return mean, std
    
    def get_action(self, state, deterministic=False):
        mean, std = self(state)
        
        if deterministic:
            return mean
        
        # 使用截断正态分布，更加稳定
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        
        # 将动作限制在[-1,1]范围内
        action = torch.tanh(x_t)
        
        return action
    
    def evaluate(self, state, action):
        mean, std = self(state)
        dist = Normal(mean, std)
        
        # 重参数化技巧提高数值稳定性
        action_probs = dist.log_prob(action)
        log_prob = action_probs.sum(dim=-1)
        entropy = dist.entropy().mean()
        
        return log_prob, entropy

class PPOCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state):
        return self.network(state)

class PPOBuffer:
    def __init__(self, state_dim, action_dim, buffer_size, device):
        self.device = device
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        
        self.ptr = 0
        self.size = 0
        self.max_size = buffer_size
        
    def add(self, state, action, reward, done, log_prob, value):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def get_all(self):
        return (
            self.states[:self.size],
            self.actions[:self.size],
            self.rewards[:self.size],
            self.dones[:self.size],
            self.log_probs[:self.size],
            self.values[:self.size]
        )
        
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0

def train_ppo(max_episodes=1000, enable_gui=False, enable_wind=False):
    """
    训练PPO算法，控制无人机悬停在(0,0,1)位置
    
    参数:
        max_episodes: 最大训练episode数量
        enable_gui: 是否启用GUI，默认为False以加速训练
        enable_wind: 是否启用随机风力干扰
    """
    # 声明使用全局变量
    global total_steps
    
    # 训练超参数
    state_dim = 18  # [x, y, z, target_x, target_y, target_z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, x_error, y_error, z_error]
    action_dim = 4   # 四个旋翼的转速
    
    # 旋翼转速参数设置
    min_rpm = 4500   # 最小转速
    max_rpm = 5500   # 最大转速
    
    # 计算动作范围
    rpm_range = max_rpm - min_rpm
    
    # PPO超参数调整
    epochs = 10
    batch_size = 128        # 增大批量大小，提升训练稳定性
    buffer_size = 4096      # 增大缓冲区大小，收集更多样本
    gamma = 0.99            # 折扣因子
    gae_lambda = 0.95       # GAE参数
    clip_ratio = 0.2        # PPO裁剪参数
    ent_coef = 0.01         # 熵系数
    vf_coef = 0.5           # 价值函数系数
    max_grad_norm = 0.5     # 梯度裁剪值
    ppo_update_iterations = 8   # 每次更新迭代次数
    
    # 学习率
    actor_lr = 3e-4
    critic_lr = 1e-3
    
    # 异步环境参数
    async_reset = True      # 启用异步重置
    update_freq = 2048      # 固定步数更新一次策略，不等待episode结束
    
    # 悬停检测参数
    velocity_threshold = 0.05  # 速度稳定阈值 (m/s)
    angular_velocity_threshold = 0.1  # 角速度稳定阈值 (rad/s)
    attitude_threshold = 0.05  # 姿态角稳定阈值 (rad)
    position_threshold = 0.1  # 位置稳定阈值 (m)
    stable_time_required = 100  # 需要保持稳定的时间步数
    
    # 限制每回合最大步数
    max_steps_per_episode = 300
    
    # 学习率调度 - 随着训练进行逐渐降低学习率
    def lr_scheduler(initial_lr, step, total_steps=1e6, min_lr=1e-5):
        # 线性学习率衰减
        return max(initial_lr * (1 - step / total_steps), min_lr)
    
    # 创建环境 - 训练时禁用GUI以提高速度
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = HoverEnv(device=device, enable_gui=enable_gui, enable_wind=enable_wind)
    
    # 打印风力设置
    if enable_wind:
        print(f"训练环境风力设置: 启用随机风力，强度范围=[{env.min_wind_strength:.2f}N, {env.max_wind_strength:.2f}N]，风向=随机")
    
    # 创建PPO模型
    actor = PPOActor(state_dim, action_dim).to(device)
    critic = PPOCritic(state_dim).to(device)
    
    # 创建优化器
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
    
    # 创建缓冲区
    buffer = PPOBuffer(state_dim, action_dim, buffer_size, device)
    
    # 训练主循环
    total_steps = 0
    next_state = env.reset()  # 在循环外先重置一次环境
    episode_reward = 0
    episode_steps = 0
    episode = 0
    
    # 跟踪训练的平均奖励
    avg_rewards = []
    best_avg_reward = -float('inf')
    
    # 稳定悬停检测
    stable_hover_time = 0
    
    # 回合结束原因跟踪
    episode_end_reason = ""
    
    # 创建输出目录
    os.makedirs('data/train_results', exist_ok=True)
    
    # 初始化数据存储
    episode_rewards = []
    position_errors = []
    control_inputs = []
    
    # 风力测试数据 - 只在训练后期或结束后记录
    wind_test_data = {
        'wind_speeds': [],
        'position_errors': [],
        'recovery_times': []
    }
    
    # 设置开始记录风力数据的episode
    start_wind_recording_episode = 600  # 从第600个episode开始记录风力数据
    
    print("开始训练...")
    if not enable_gui:
        print("无GUI模式，训练速度将大幅提升")
    
    # 打印表头
    if enable_wind:
        header = "回合  |  步数  |   奖励   |  平均奖励  |   风力   |   风向   | 结束原因"
    else:
        header = "回合  |  步数  |   奖励   |  平均奖励  | 结束原因"
    divider = "-" * 100
    print(divider)
    print(header)
    print(divider)
    
    while episode < max_episodes:  # 总episode数限制
        state = next_state
        
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        
        # 预测动作和价值
        with torch.no_grad():
            action_mean, action_std = actor(state_tensor)
            value = critic(state_tensor)
            action_dist = Normal(action_mean, action_std)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)
        
        # 将动作从[-1,1]范围映射到[min_rpm, max_rpm]
        actions_np = action.cpu().numpy()
        
        # 直接映射到转速范围
        rpm_action = min_rpm + (actions_np + 1) * 0.5 * rpm_range
        
        # 限制在合法范围内
        rpm_action = np.clip(rpm_action, min_rpm, max_rpm)
        
        # 与环境交互
        next_state, reward, done, _ = env.step(rpm_action)
        episode_reward += reward
        episode_steps += 1
        total_steps += 1
        
        # 检查是否达到最大步数
        if episode_steps >= max_steps_per_episode:
            done = True
            episode_end_reason = f"达到最大步数限制 ({max_steps_per_episode}步)"
        
        # 获取速度和姿态信息
        lin_vel = np.array([next_state[9], next_state[10], next_state[11]])  # 线速度
        ang_vel = np.array([next_state[12], next_state[13], next_state[14]])  # 角速度
        roll, pitch, yaw = next_state[6], next_state[7], next_state[8]  # 姿态角 [roll, pitch, yaw]
        
        # 位置误差
        position_error = np.array([next_state[15], next_state[16], next_state[17]])
        distance_to_target = np.linalg.norm(position_error)
        
        # 检查无人机是否稳定悬停
        lin_vel_stable = np.all(np.abs(lin_vel) < velocity_threshold)
        ang_vel_stable = np.all(np.abs(ang_vel) < angular_velocity_threshold)
        attitude_stable = (abs(roll) < attitude_threshold and abs(pitch) < attitude_threshold)
        position_stable = distance_to_target < position_threshold
        
        if lin_vel_stable and ang_vel_stable and attitude_stable and position_stable:
            stable_hover_time += 1
        else:
            stable_hover_time = 0
            
        # 如果持续稳定悬停一段时间，提前结束当前episode
        if stable_hover_time >= stable_time_required:
            episode_end_reason = f"稳定悬停达到阈值 ({stable_time_required}步)"
            done = True
        
        # 存储经验
        buffer.add(
            state_tensor,
            action,
            reward,
            float(done),
            log_prob.item(),
            value.item()
        )
        
        # 当收集到指定步数或episode结束时更新策略
        update_condition = buffer.size >= update_freq or done
        
        if update_condition and buffer.size >= batch_size:  # 确保有足够数据进行更新
            # 计算最后一个状态的价值（用于GAE计算）
            if not done:
                with torch.no_grad():
                    final_value = critic(torch.tensor(next_state, dtype=torch.float32, device=device)).item()
            else:
                final_value = 0.0
            
            # 调整学习率
            current_actor_lr = lr_scheduler(actor_lr, total_steps)
            current_critic_lr = lr_scheduler(critic_lr, total_steps)
            for param_group in actor_optimizer.param_groups:
                param_group['lr'] = current_actor_lr
            for param_group in critic_optimizer.param_groups:
                param_group['lr'] = current_critic_lr
                
            # 执行PPO更新
            update_start_time = time.time()
            
            # 获取数据
            states, actions, rewards, dones, old_log_probs, values = buffer.get_all()
            
            # 计算优势估计和回报
            returns, advantages = compute_gae(
                rewards, values, dones, final_value, gamma, gae_lambda
            )
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            for _ in range(ppo_update_iterations):
                # 分批次训练，减少内存占用，使用随机打乱增加样本多样性
                indices = torch.randperm(buffer.size)
                for start_idx in range(0, buffer.size, batch_size):
                    # 批量数据处理
                    end_idx = min(start_idx + batch_size, buffer.size)
                    batch_indices = indices[start_idx:end_idx]
                    
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    
                    # 评估当前策略
                    new_log_probs, entropy = actor.evaluate(batch_states, batch_actions)
                    new_values = critic(batch_states).squeeze(-1)
                    
                    # 策略损失（PPO-Clip目标）
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    obj = ratio * batch_advantages
                    obj_clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_advantages
                    policy_loss = -torch.min(obj, obj_clipped).mean()
                    
                    # 价值损失
                    value_loss = 0.5 * ((new_values - batch_returns) ** 2).mean()
                    
                    # 熵奖励
                    entropy_loss = -entropy
                    
                    # 总损失
                    loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
                    
                    # 更新策略和价值网络
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    loss.backward()
                    
                    # 梯度裁剪
                    nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                    nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                    
                    actor_optimizer.step()
                    critic_optimizer.step()
            
            # 清空缓冲区，而不是创建新的缓冲区
            buffer.clear()
        
        # 如果episode结束，则重置环境并记录统计信息
        if done:
            # 如果还没有设置结束原因，说明是环境内部检测到越界等原因
            if not episode_end_reason:
                # 获取高度和XY位置
                x, y, z = next_state[0], next_state[1], next_state[2]
                if z < env.min_height or z > env.max_height:
                    episode_end_reason = f"高度超出范围 ({z:.2f})"
                elif abs(x) > env.xy_boundary or abs(y) > env.xy_boundary:
                    episode_end_reason = f"水平位置超出范围 ({x:.2f}, {y:.2f})"
                else:
                    episode_end_reason = "环境内部终止条件"
            
            # 记录episode数据
            episode_rewards.append(episode_reward)
            position_errors.append(np.linalg.norm(position_error))
            
            # 只在训练后期记录风力测试数据
            if enable_wind and episode >= start_wind_recording_episode:
                wind_test_data['wind_speeds'].append(env.wind_strength)
                wind_test_data['position_errors'].append(np.linalg.norm(position_error))
                wind_test_data['recovery_times'].append(episode_steps * env.dt)
            
            # 记录平均奖励
            avg_rewards.append(episode_reward)
            if len(avg_rewards) > 10:
                avg_rewards.pop(0)
            current_avg_reward = sum(avg_rewards) / len(avg_rewards)
            
            # 每100个episode保存一次数据
            if episode % 100 == 0:
                try:
                    # 保存训练奖励数据
                    rewards_data = np.array(episode_rewards)
                    np.savetxt('outputs/plots/training_rewards.csv', rewards_data,
                              delimiter=',', header='episode_reward', comments='')
                    print(f"✓ 训练奖励数据已保存到 training_rewards.csv (共{len(rewards_data)}条记录)")
                    
                    # 保存位置误差数据
                    errors_data = np.array(position_errors)
                    np.savetxt('outputs/plots/position_errors.csv', errors_data,
                              delimiter=',', header='position_error', comments='')
                    print(f"✓ 位置误差数据已保存到 position_errors.csv (共{len(errors_data)}条记录)")
                    
                    # 保存控制输入数据
                    if control_inputs:
                        inputs_data = np.array(control_inputs)
                        np.savetxt('outputs/plots/control_inputs.csv', inputs_data,
                                  delimiter=',', header='motor1,motor2,motor3,motor4', comments='')
                        print(f"✓ 控制输入数据已保存到 control_inputs.csv (共{len(inputs_data)}条记录)")
                    
                    # 如果启用了风力，且已经过了开始记录风力的episode，保存风力适应数据
                    if enable_wind and episode >= start_wind_recording_episode and wind_test_data['wind_speeds']:
                        # 按风速分组计算平均误差和恢复时间
                        unique_wind_speeds = np.unique(wind_test_data['wind_speeds'])
                        avg_errors = []
                        avg_recovery_times = []
                        
                        for wind_speed in unique_wind_speeds:
                            # 找出该风速下的所有数据索引
                            indices = [i for i, speed in enumerate(wind_test_data['wind_speeds']) if speed == wind_speed]
                            # 计算平均误差和恢复时间
                            avg_errors.append(np.mean([wind_test_data['position_errors'][i] for i in indices]))
                            avg_recovery_times.append(np.mean([wind_test_data['recovery_times'][i] for i in indices]))
                        
                        # 保存按风速分组的平均数据
                        wind_data = np.column_stack((
                            unique_wind_speeds,
                            avg_errors,
                            avg_recovery_times
                        ))
                        np.savetxt('outputs/plots/wind_adaptation.csv', wind_data,
                                  delimiter=',', header='wind_speed,position_error,recovery_time', comments='')
                        print(f"✓ 风力适应数据已保存到 wind_adaptation.csv")
                        print(f"  - 测试的风速范围: {min(unique_wind_speeds):.2f}N 到 {max(unique_wind_speeds):.2f}N")
                        print(f"  - 平均位置误差: {np.mean(avg_errors):.3f}m")
                        print(f"  - 平均恢复时间: {np.mean(avg_recovery_times):.2f}s")
                    
                    print(f"\n第 {episode} 回合数据已保存完成")
                except Exception as e:
                    print(f"保存数据时出错: {str(e)}")
            
            # 启用异步重置，在前一个episode结束后立即开始下一个
            if async_reset:
                # 使用异步方式重置环境
                next_state = env.reset()
            
            # 格式化输出回合信息
            if enable_wind:
                print(f"回合: {episode:5d} | 步数: {episode_steps:5d} | 奖励: {episode_reward:8.2f} | 平均奖励: {current_avg_reward:8.2f} | 风力: {env.wind_strength:7.2f}N | 风向: {env.wind_direction_deg:8.1f}° | 结束原因: {episode_end_reason}")
            else:
                print(f"回合: {episode:5d} | 步数: {episode_steps:5d} | 奖励: {episode_reward:8.2f} | 平均奖励: {current_avg_reward:8.2f} | 结束原因: {episode_end_reason}")
            
            episode += 1
            episode_reward = 0
            episode_steps = 0
            stable_hover_time = 0  # 重置稳定悬停时间
            episode_end_reason = ""  # 重置结束原因
            
            # 如果非异步重置，在这里重置环境
            if not async_reset:
                next_state = env.reset()
            
            # 仅保存最佳模型
            if current_avg_reward > best_avg_reward:
                best_avg_reward = current_avg_reward
                model_path = f"models/ppo/quad_hover_ppo_best_model.pt"
                
                torch.save({
                    'actor': actor.state_dict(),
                    'critic': critic.state_dict(),
                    'actor_optimizer': actor_optimizer.state_dict(),
                    'critic_optimizer': critic_optimizer.state_dict(),
                    'episode': episode,
                    'total_steps': total_steps,
                    'avg_reward': current_avg_reward
                }, model_path)
                print(f"最佳模型已保存: {model_path}, 平均奖励: {current_avg_reward:.2f}")
    
    # 训练结束，保存最终数据
    try:
        # 保存最终训练奖励数据
        rewards_data = np.array(episode_rewards)
        np.savetxt('outputs/plots/training_rewards.csv', rewards_data,
                  delimiter=',', header='episode_reward', comments='')
        print(f"✓ 最终训练奖励数据已保存到 training_rewards.csv (共{len(rewards_data)}条记录)")
        
        # 保存最终位置误差数据
        errors_data = np.array(position_errors)
        np.savetxt('outputs/plots/position_errors.csv', errors_data,
                  delimiter=',', header='position_error', comments='')
        print(f"✓ 最终位置误差数据已保存到 position_errors.csv (共{len(errors_data)}条记录)")
        
        # 保存最终控制输入数据
        if control_inputs:
            inputs_data = np.array(control_inputs)
            np.savetxt('outputs/plots/control_inputs.csv', inputs_data,
                      delimiter=',', header='motor1,motor2,motor3,motor4', comments='')
            print(f"✓ 最终控制输入数据已保存到 control_inputs.csv (共{len(inputs_data)}条记录)")
        
        # 如果启用了风力，保存最终风力适应数据
        if enable_wind and wind_test_data['wind_speeds']:
            # 按风速分组计算平均误差和恢复时间
            unique_wind_speeds = np.unique(wind_test_data['wind_speeds'])
            avg_errors = []
            avg_recovery_times = []
            
            for wind_speed in unique_wind_speeds:
                # 找出该风速下的所有数据索引
                indices = [i for i, speed in enumerate(wind_test_data['wind_speeds']) if speed == wind_speed]
                # 计算平均误差和恢复时间
                avg_errors.append(np.mean([wind_test_data['position_errors'][i] for i in indices]))
                avg_recovery_times.append(np.mean([wind_test_data['recovery_times'][i] for i in indices]))
            
            # 保存按风速分组的平均数据
            wind_data = np.column_stack((
                unique_wind_speeds,
                avg_errors,
                avg_recovery_times
            ))
            np.savetxt('outputs/plots/wind_adaptation.csv', wind_data,
                      delimiter=',', header='wind_speed,position_error,recovery_time', comments='')
            print("\n最终风力适应数据统计:")
            print(f"✓ 风力适应数据已保存到 wind_adaptation.csv")
            print(f"  - 测试的风速范围: {min(unique_wind_speeds):.2f}N 到 {max(unique_wind_speeds):.2f}N")
            print(f"  - 平均位置误差: {np.mean(avg_errors):.3f}m")
            print(f"  - 平均恢复时间: {np.mean(avg_recovery_times):.2f}s")
        
        print("\n训练结束，所有数据已保存完成")
    except Exception as e:
        print(f"保存最终数据时出错: {str(e)}")
    
    # 关闭环境
    env.close()

def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """计算广义优势估计 (GAE) 和折扣累积回报"""
    rewards = torch.as_tensor(rewards, dtype=torch.float32).clone().detach()
    values = torch.as_tensor(values, dtype=torch.float32).clone().detach()
    dones = torch.as_tensor(dones, dtype=torch.float32).clone().detach()
    
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    
    last_gae_lam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
            
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam
        
    returns = advantages + values
    
    # 标准化优势估计
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return returns, advantages

def evaluate_model(model_path=None, num_episodes=5, enable_gui=True, enable_wind=False):
    """
    评估训练好的模型在悬停任务上的表现
    
    参数:
        model_path: 模型保存的路径，若为None则默认使用最终模型
        num_episodes: 测试的episode数量
        enable_gui: 是否启用GUI，默认为True进行可视化
        enable_wind: 是否启用随机风力干扰
    """
    # 如果未指定模型路径，默认使用最终模型
    if model_path is None:
        model_path = "models/ppo/quad_hover_ppo_final_model.pt"
    
    # 设置参数
    state_dim = 18  # 与训练时相同
    action_dim = 4   # 与训练时相同
    min_rpm = 4500
    max_rpm = 5500
    rpm_range = max_rpm - min_rpm
    
    # 创建环境 - 评估时默认启用GUI进行可视化
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = HoverEnv(device=device, enable_gui=enable_gui, enable_wind=enable_wind)
    
    # 打印风力设置
    if enable_wind:
        print(f"评估环境风力设置: 启用随机风力，强度范围=[{env.min_wind_strength:.2f}N, {env.max_wind_strength:.2f}N]，风向=随机")
        print(f"风力更新频率: 每300步更新一次")
    else:
        print("评估环境风力设置: 无风干扰")
    
    # 加载模型
    actor = PPOActor(state_dim, action_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()
    
    print(f"加载模型: {model_path}")
    if enable_gui:
        print("GUI模式已启用，可以观察无人机飞行状态")
    
    episode_rewards = []
    episode_durations = []
    hover_success = 0  # 成功悬停的次数
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        # 记录无人机轨迹
        hover_time = 0
        stable_hover_time = 0  # 稳定悬停的时间
        hover_success_this_episode = False  # 标记当前回合是否已达成悬停成功
        
        # 记录风力变化
        wind_change_count = 0  # 风力变化次数
        
        print(f"\n=============== 开始评估回合 {episode+1} ===============")
        # 打印当前回合的风力信息
        if enable_wind:
            print(f"初始风力: 强度={env.wind_strength:.2f}N, 风向={env.wind_direction_deg:.1f}°")
        
        while not done:
            # 每300步更新一次风力
            if enable_wind and steps > 0 and steps % 300 == 0:
                # 保存旧风力信息
                old_strength = env.wind_strength
                old_direction = env.wind_direction_deg
                
                # 重置风力
                env._reset_wind()
                wind_change_count += 1
                
                # 打印风力变化信息
                print(f"\n--- 步数: {steps}，风力已更新 (第{wind_change_count}次) ---")
                print(f"旧风力: 强度={old_strength:.2f}N, 风向={old_direction:.1f}°")
                print(f"新风力: 强度={env.wind_strength:.2f}N, 风向={env.wind_direction_deg:.1f}°")
            
            # 转换状态为tensor
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            
            # 使用模型预测动作（确定性模式）
            with torch.no_grad():
                action = actor.get_action(state_tensor, deterministic=True)
            
            # 将动作从[-1,1]范围映射到[min_rpm, max_rpm]
            actions_np = action.cpu().numpy()
            rpm_action = min_rpm + (actions_np + 1) * 0.5 * rpm_range
            rpm_action = np.clip(rpm_action, min_rpm, max_rpm)
            
            # 执行动作并获取下一个状态
            next_state, reward, done, _ = env.step(rpm_action)
            
            # 获取当前位置
            pos = next_state[0:3]  # [x, y, z]
            lin_vel = next_state[9:12]  # [vx, vy, vz]
            ang_vel = next_state[12:15]  # [wx, wy, wz]
            roll, pitch, yaw = next_state[6:9]  # [roll, pitch, yaw]
            position_error = next_state[15:18]  # [x_error, y_error, z_error]
            
            # 实时覆盖输出无人机状态信息，如果有风力，则显示当前风力信息
            wind_info = ""
            if enable_wind:
                # 计算当前实际风力和方向（考虑变化）
                curr_wind_x = env.wind_force_x
                curr_wind_y = env.wind_force_y
                curr_wind_strength = np.sqrt(curr_wind_x**2 + curr_wind_y**2)
                curr_wind_dir = np.degrees(np.arctan2(curr_wind_y, curr_wind_x))
                wind_info = f" | 风力: {curr_wind_strength:.2f}N, 风向: {curr_wind_dir:.1f}°"
            
            print(f"\r步数: {steps:3d} | 位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | "
                  f"速度: [{lin_vel[0]:.2f}, {lin_vel[1]:.2f}, {lin_vel[2]:.2f}] | "
                  f"姿态: [{roll:.2f}, {pitch:.2f}, {yaw:.2f}] | "
                  f"误差: {np.linalg.norm(position_error):.3f}{wind_info}", end="")
            
            # 计算当前位置与目标位置的距离
            distance_to_target = np.linalg.norm(position_error)
            
            # 检查是否接近目标位置
            if distance_to_target < 0.1:  # position_threshold = 0.1
                hover_time += 1
            else:
                hover_time = 0
            
            # 检查无人机是否稳定悬停（速度和姿态角都很小）
            lin_vel_stable = np.all(np.abs(lin_vel) < 0.05)  # velocity_threshold = 0.05 m/s
            ang_vel_stable = np.all(np.abs(ang_vel) < 0.1)   # angular_velocity_threshold = 0.1 rad/s
            attitude_stable = (abs(roll) < 0.05 and          # attitude_threshold = 0.05 rad
                               abs(pitch) < 0.05)
            
            if lin_vel_stable and ang_vel_stable and attitude_stable and distance_to_target < 0.1:
                stable_hover_time += 1
                # 打印稳定悬停信息，但不结束评估
                if stable_hover_time == 100 and not hover_success_this_episode:  # stable_time_required = 100
                    print(f"\n检测到稳定悬停，继续评估观察性能")
                    hover_success_this_episode = True
            else:
                stable_hover_time = 0
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
            steps += 1
            
            # 防止无限循环，设置最大步数
            if steps >= 5000:
                done = True
        
        # 计算悬停成功
        if hover_success_this_episode or hover_time >= 300:
            hover_success += 1
        
        # 计算飞行持续时间 (以秒为单位)
        flight_duration = steps * env.dt
        
        # 打印回合总结，加入风力信息
        wind_summary = ""
        if enable_wind:
            wind_summary = f", 风力变化次数: {wind_change_count}, 强度范围: {env.min_wind_strength:.2f}N-{env.max_wind_strength:.2f}N"
        
        print(f"\n回合 {episode+1} 评估结果 | 飞行持续时间: {flight_duration:.2f}秒 | 总奖励: {episode_reward:.2f}{wind_summary}")
    
    # 打印评估结果
    avg_reward = sum(episode_rewards) / num_episodes
    avg_duration = sum(episode_durations) / num_episodes
    success_rate = hover_success / num_episodes * 100
    
    print("\n===== 评估结果 =====")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均飞行时间: {avg_duration:.2f}秒")
    print(f"悬停成功率: {success_rate:.1f}%")
    print(f"风力设置: {'启用' if enable_wind else '禁用'}")
    
    # 关闭环境
    env.close()
    return success_rate

def test_wind_adaptation(env, actor, device, min_rpm, max_rpm, num_tests=5):
    """测试模型在不同风力条件下的适应能力"""
    # 创建输出目录
    os.makedirs('outputs/plots', exist_ok=True)
    
    # 测试不同的风速
    wind_speeds = [0.2, 0.3, 0.4, 0.5]
    wind_errors = []
    wind_recovery_times = []
    
    print("\n开始风力适应测试...")
    print("风速(N) | 最大误差(m) | 恢复时间(s)")
    print("-" * 40)
    
    for wind_speed in wind_speeds:
        env.wind_strength = wind_speed
        
        episode_errors = []
        episode_recovery_times = []
        
        for test in range(num_tests):
            state = env.reset()
            max_error = 0
            recovery_start = None
            
            for t in range(200):  # 测试200步
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action = actor.get_action(state_tensor, deterministic=True)
                rpm_action = min_rpm + (action.cpu().numpy() + 1) * 0.5 * (max_rpm - min_rpm)
                state, _, done, _ = env.step(rpm_action)
                
                error = np.linalg.norm(state[:3] - env.target_position)
                max_error = max(max_error, error)
                
                if error < 0.1 and recovery_start is None:  # 误差小于0.1m时开始计时
                    recovery_start = t
                
                if done:
                    break
            
            episode_errors.append(max_error)
            episode_recovery_times.append((recovery_start or t) * env.dt)
        
        # 计算该风速下的平均误差和恢复时间
        avg_error = np.mean(episode_errors)
        avg_recovery_time = np.mean(episode_recovery_times)
        wind_errors.append(avg_error)
        wind_recovery_times.append(avg_recovery_time)
        
        # 打印当前风速的测试结果
        print(f"风速: {wind_speed:4.1f}N | 最大误差: {avg_error:8.3f}m | 恢复时间: {avg_recovery_time:8.2f}s")
    
    # 保存数据到CSV文件
    wind_data = np.column_stack((wind_speeds, wind_errors, wind_recovery_times))
    np.savetxt('outputs/plots/wind_adaptation_data.csv', wind_data, 
               delimiter=',', header='wind_speed,position_error,recovery_time',
               comments='')
    
    print("\n风力适应测试完成")
    print(f"平均位置误差: {np.mean(wind_errors):.3f}m")
    print(f"平均恢复时间: {np.mean(wind_recovery_times):.2f}s")
    print("测试数据已保存到 wind_adaptation_data.csv")
    
    return np.mean(wind_errors), np.mean(wind_recovery_times)

if __name__ == "__main__":
    import sys
    
    # 默认参数
    enable_wind = False
    
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        # 评估模式
        model_path = None  # 默认使用最终模型
        
        # 解析评估参数
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--model" and i + 1 < len(sys.argv):
                model_path = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--wind":
                enable_wind = True
                i += 1
            else:
                i += 1
                
        evaluate_model(model_path, enable_gui=True, enable_wind=enable_wind)
    else:
        # 训练模式
        max_episodes = 2000  # 默认最大episode数
        enable_gui = False   # 默认训练时关闭GUI以加速训练
        
        # 解析命令行参数
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == "--gui":
                enable_gui = True
                print("启用GUI模式进行训练（可能会降低训练速度）")
                i += 1
            elif sys.argv[i] == "--wind":
                enable_wind = True
                i += 1
            else:
                try:
                    max_episodes = int(sys.argv[i])
                    i += 1
                except ValueError:
                    print("参数必须是整数")
                    i += 1
        
        train_ppo(max_episodes, enable_gui=enable_gui, enable_wind=enable_wind) 

        # 使用示例:
        # 训练无风环境: python drone_PPO.py 1000
        # 训练有风环境: python drone_PPO.py 2000 --wind
        # 评估无风环境: python drone_PPO.py evaluate 
        # 评估有风环境: python drone_PPO.py evaluate --wind