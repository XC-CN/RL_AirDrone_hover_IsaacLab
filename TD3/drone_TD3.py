import torch
import numpy as np
import argparse
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from isaaclab.app import AppLauncher

def rpm_to_force(rpm, k_f=1e-6):
    """将转速(RPM)转换为推力
    
    参数:
        rpm: 电机转速（每分钟转数）
        k_f: 推力系数
    
    返回:
        计算得到的推力值
    """
    rad_s = rpm * 2 * 3.14159 / 60.0  # 转换为弧度/秒
    return k_f * rad_s**2  # 推力与角速度的平方成正比

class HoverEnv:
    """无人机悬停环境类，用于模拟四旋翼无人机的悬停任务"""
    
    def __init__(self, device="cpu", headless=True, verbose=False):
        """初始化悬停环境
        
        参数:
            device: 运行设备，可以是'cpu'或'cuda'
            headless: 是否使用无界面模式
            verbose: 是否输出详细信息
        """
        self.device = device
        self.min_rpm = 4900  # 最小电机转速 - 扩大范围
        self.max_rpm = 5400  # 最大电机转速 - 扩大范围
        self._verbose = verbose
        self.current_step = 0
        self.hover_time = 0  # 记录悬停时间
        
        # 计算平衡悬停所需的估计转速
        self.hover_rpm = 5100 # 悬停所需的估计电机转速，可以通过实验调整
        
        # 初始化Isaac模拟器
        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args_cli = parser.parse_args(args=[])
        args_cli.headless = headless  # 设置无界面模式
        self.app_launcher = AppLauncher(args_cli)
        self.simulation_app = self.app_launcher.app

        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab_assets import CRAZYFLIE_CFG

        # 配置模拟环境
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=device)  # 设置时间步长和设备
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view(eye=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.5])  # 设置相机视角

        # 创建地面
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/defaultGroundPlane", ground)

        # 添加光源
        light = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
        light.func("/World/Light", light)

        # 配置无人机模型
        robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Crazyflie")
        robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
        self.robot = Articulation(robot_cfg)

        # 重置模拟环境
        self.sim.reset()
        self.prop_ids = self.robot.find_bodies("m.*_prop")[0]  # 获取螺旋桨的ID
        self.dt = self.sim.get_physics_dt()  # 获取物理时间步长

    def reset(self, *, seed=None, options=None):
        """重置环境到初始状态
        
        参数:
            seed: 随机种子
            options: 重置选项
            
        返回:
            初始状态和额外信息
        """
        # 重置计步器和悬停时间
        self.current_step = 0
        self.hover_time = 0
            
        # 设置随机种子（如果提供）
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        # 重置关节状态和位置
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)
        self.robot.write_root_pose_to_sim(self.robot.data.default_root_state[:, :7])
        self.robot.write_root_velocity_to_sim(self.robot.data.default_root_state[:, 7:])
        self.robot.reset()
        
        # 设置初始四个电机转速相等
        initial_rpm = self.hover_rpm  # 初始转速设置为悬停估计值
        forces = torch.zeros(1, 4, 3, device=self.device)
        torques = torch.zeros_like(forces)
        
        # 为四个电机设置相同的转速
        for i in range(4):
            thrust = rpm_to_force(initial_rpm)
            forces[0, i, 2] = thrust  # 力作用在z轴方向

        # 施加外力和力矩
        self.robot.set_external_force_and_torque(forces, torques, body_ids=self.prop_ids)
        self.robot.write_data_to_sim()
        
        # 执行一步更新以应用初始旋翼速度
        self.sim.step()
        self.robot.update(self.dt)
        
        # 返回状态和额外信息
        state = self._get_state()
        return state, {"initial_rpm": initial_rpm}

    def _get_state(self):
        """获取当前环境状态
        
        返回:
            包含位置、速度、倾角等信息的状态数组
        """
        # 获取位置信息
        pos = self.robot.data.root_pos_w[0].cpu().numpy()  # [x, y, z]
        x, y, z = pos[0], pos[1], pos[2]  # 获取x, y, z位置坐标
        
        # 获取速度信息
        vel = self.robot.data.root_vel_w[0, :3].cpu().numpy()  # [vx, vy, vz]
        vx, vy, vz = vel[0], vel[1], vel[2]  # 获取x, y, z方向速度
        
        # 获取角速度信息
        angular_vel = self.robot.data.root_vel_w[0, 3:6].cpu().numpy()  # 角速度
        
        # 获取四元数 - 使用正确的root_quat_w变量
        quat = self.robot.data.root_quat_w[0]  # 获取四元数 [qw, qx, qy, qz]
        
        # 计算俯仰角和横滚角
        qw, qx, qy, qz = quat.cpu().numpy()
        # 简化的欧拉角计算（使用四元数估算倾角）
        roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = np.arcsin(2.0 * (qw * qy - qz * qx))
        
        # 计算总倾角（俯仰角和横滚角的平方和的平方根）
        tilt_angle = np.sqrt(roll**2 + pitch**2)
        
        # 返回扩展的12维状态：
        # [x, y, z, vx, vy, vz, 总倾角, 横滚角, 俯仰角, x角速度, y角速度, z角速度]
        return np.array([x, y, z, vx, vy, vz, tilt_angle, roll, pitch, 
                        angular_vel[0], angular_vel[1], angular_vel[2]], dtype=np.float32)

    def step(self, rpms):
        """执行一步模拟
        
        参数:
            rpms: 四个电机的转速
            
        返回:
            新状态、奖励、是否结束、终止原因、额外信息
        """
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

        # 执行多个物理模拟步骤
        for _ in range(4):
            self.sim.step()
            self.robot.update(self.dt)

        # 获取新状态并计算奖励
        state = self._get_state()
        
        # 解析扩展状态
        x, y, z = state[0], state[1], state[2]         # 位置
        vx, vy, vz = state[3], state[4], state[5]      # 速度
        tilt_angle = state[6]                         # 总倾角
        roll, pitch = state[7], state[8]              # 横滚角和俯仰角
        
        # 目标高度和理想状态
        target_height = 1.0
        
        # 计算水平位置偏移量（相对于起点）
        horizontal_offset = np.sqrt(x**2 + y**2)
        
        # 使用高度和倾角计算奖励 - 增加对高度接近目标的更强激励
        height_diff = abs(z - target_height)
        
        # 非线性奖励函数，接近目标高度时奖励增长更快
        if height_diff < 0.15:  # 非常接近目标高度
            height_reward = 2.0 - 10.0 * height_diff  # 接近目标时较大奖励
        else:
            height_reward = -2.0 * height_diff  # 远离目标时较大惩罚
            
        # 速度奖励：悬停需要垂直速度接近0
        if abs(vz) < 0.08:  # 速度非常小
            velocity_reward = 1.0  # 给予额外奖励
        else:
            velocity_reward = -0.5 * abs(vz)  # 惩罚速度
            
        # 水平位置和速度奖励/惩罚
        # 水平位置偏移惩罚
        position_penalty = -1.0 * horizontal_offset
        
        # 水平速度惩罚
        horizontal_velocity = np.sqrt(vx**2 + vy**2)
        velocity_penalty = -0.5 * horizontal_velocity
            
        # 倾角惩罚 - 使用指数惩罚
        tilt_penalty = -3.0 * (tilt_angle**2)  # 二次方惩罚，更严厉地惩罚大倾角
        
        # 悬停成功奖励：如果所有条件都很理想，给予额外奖励鼓励保持该状态
        hover_success = False
        # 使用更宽松的稳定悬停条件，让初期学习更容易
        if (height_diff < 0.15 and abs(vz) < 0.08 and tilt_angle < 0.15 and 
            horizontal_offset < 0.15 and horizontal_velocity < 0.08):
            hover_success = True
            hover_bonus = 3.0  # 悬停成功额外奖励
            self.hover_time += 1  # 增加悬停时间计数
        else:
            hover_bonus = 0.0
            self.hover_time = 0  # 重置悬停时间计数
        
        # 整合所有奖励 - 不再包含电机协调性相关奖励
        reward = height_reward + velocity_reward + tilt_penalty + hover_bonus + position_penalty + velocity_penalty
        
        # 成功悬停一定时间后提前终止（加快训练）
        success_termination = self.hover_time >= 100  # 连续悬停100步视为成功
        
        # 扩大无人机活动空间，给予更多探索机会
        # 1. 增加倾角容忍度，让无人机有更大的运动空间
        max_tilt_angle = 0.5  # 增加到0.5（约30度）
        
        # 2. 扩大高度容忍范围
        min_height = 0.3   # 降低到0.3米
        max_height = 1.8   # 提高到1.8米
        
        # 3. 增加速度容忍度
        max_velocity = 1.5  # 垂直速度最大值增加到1.5米/秒
        
        # 4. 增加水平距离容忍度
        max_horizontal_offset = 1.5  # 最大水平偏移距离（米）
        
        # 使用更严格的终止条件，但不再使用步数限制，除非检测到稳定悬停
        done = (
            tilt_angle > max_tilt_angle or      # 倾角过大
            z < min_height or                  # 高度过低
            z > max_height or                  # 高度过高
            abs(vz) > max_velocity or          # 垂直速度过大
            horizontal_offset > max_horizontal_offset or  # 水平偏移过大
            success_termination                # 成功悬停足够长时间
        )
        
        # 创建信息字典
        info_dict = {
            'step': self.current_step,
            'hover_time': self.hover_time,
            'success': success_termination,
            'rpms': rpms.copy().tolist(),  # 记录当前使用的RPM值
            'stable_hover': hover_success,  # 是否当前帧处于稳定悬停
            'height_diff': height_diff,    # 高度差异
            'tilt': tilt_angle,            # 倾角
            'velocity': vz,                # 垂直速度
            'horizontal_offset': horizontal_offset,  # 水平偏移
            'horizontal_velocity': horizontal_velocity  # 水平速度
        }
        
        # 在测试模式下或显式要求时输出状态信息
        if self._verbose:
            status = "成功悬停中" if hover_success else "调整中"
            print(f"\r高度: {z:.2f}, 速度: {vz:.2f}, 倾角: {tilt_angle:.2f}, 水平偏移: {horizontal_offset:.2f}, 状态: {status}, 步数: {self.current_step}, 稳定悬停: {self.hover_time}步", end="")
        
        # 返回更新后的信息，兼容Gymnasium
        return state, reward, done, info_dict

    def close(self):
        """关闭模拟器"""
        self.simulation_app.close()

class IsaacDroneEnv(gym.Env):
    """Gym环境适配器，使IsaacLab的悬停环境符合gym.Env接口"""
    
    def __init__(self, env=None):
        """初始化环境适配器
        
        参数:
            env: 已有的HoverEnv实例，如果为None则创建新实例
        """
        super(IsaacDroneEnv, self).__init__()
        
        # 创建或使用提供的环境
        self.env = env if env is not None else HoverEnv(headless=True)
        
        # 定义动作空间（四个电机的相对转速，范围[-1, 1]）
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0, 
            shape=(4,),
            dtype=np.float32
        )
        
        # 定义观察空间（扩展为12维状态空间）
        # [x, y, z, vx, vy, vz, 总倾角, 横滚角, 俯仰角, x角速度, y角速度, z角速度]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, 0, -np.inf, -np.inf, -np.inf, 0, -np.pi, -np.pi/2, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.pi/2, np.pi, np.pi/2, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        
        # 电机基础转速和范围
        self.base_rpm = 5150  # 基础转速（悬停所需的大致转速）
        self.rpm_range = 250  # 转速可调范围 - 增加范围从150到250
        
        # 添加电机历史记录
        self.motor_history = []
        self.max_history = 5  # 保存最近5次电机值
        self.prev_action = np.zeros(4)  # 上一次的动作
        
    def reset(self, *, seed=None, options=None):
        """重置环境
        
        参数:
            seed: 随机种子
            options: 重置选项
            
        返回:
            初始观察和额外信息
        """
        # 重置电机历史
        self.motor_history = []
        
        # 将上一动作初始化为0，这样所有电机转速将等于基础转速
        self.prev_action = np.zeros(4)
        
        state, info = self.env.reset(seed=seed, options=options)
        
        # 更新基础转速，确保与HoverEnv一致
        if hasattr(self.env, 'hover_rpm'):
            self.base_rpm = self.env.hover_rpm
            
        return state, info
        
    def step(self, action):
        """执行动作并返回结果
        
        参数:
            action: 动作向量，范围[-1, 1]，表示四个电机的相对转速
            
        返回:
            观察、奖励、是否结束、是否截断、信息
        """
        # 计算动作变化，并添加平滑处理以防止电机突变
        action_diff = action - self.prev_action
        
        # 如果动作变化过大，添加平滑限制（可选）
        max_change = 0.2  # 最大允许的单步变化幅度
        if np.any(np.abs(action_diff) > max_change):
            # 限制变化率
            action_diff = np.clip(action_diff, -max_change, max_change)
            smoothed_action = self.prev_action + action_diff
        else:
            smoothed_action = action
        
        # 获取当前状态，计算偏移校正量
        current_state = self._get_state() if hasattr(self, '_get_state') else self.env._get_state()
        x, y = current_state[0], current_state[1]  # 获取当前水平位置
        
        # 添加校正逻辑 - 如果无人机偏离中心位置，我们增加校正力
        # 偏移大小
        offset = np.sqrt(x*x + y*y)
        
        if offset > 0.1:  # 只在偏移较大时进行校正
            # 计算校正方向 - 与当前偏移方向相反
            correction_x = -x / offset if abs(x) > 0.05 else 0
            correction_y = -y / offset if abs(y) > 0.05 else 0
            
            # 根据无人机坐标系和电机布局应用校正
            # 假设：电机0在前左，电机1在前右，电机2在后左，电机3在后右
            
            # X偏移校正 - 影响前后对称的电机对
            if abs(correction_x) > 0.05:
                # 前倾或后倾来修正X方向偏移
                smoothed_action[0] += 0.1 * correction_x  # 前左电机
                smoothed_action[1] += 0.1 * correction_x  # 前右电机
                smoothed_action[2] -= 0.1 * correction_x  # 后左电机
                smoothed_action[3] -= 0.1 * correction_x  # 后右电机
            
            # Y偏移校正 - 影响左右对称的电机对
            if abs(correction_y) > 0.05:
                # 左倾或右倾来修正Y方向偏移
                smoothed_action[0] += 0.1 * correction_y  # 前左电机
                smoothed_action[2] += 0.1 * correction_y  # 后左电机
                smoothed_action[1] -= 0.1 * correction_y  # 前右电机
                smoothed_action[3] -= 0.1 * correction_y  # 后右电机
            
            # 再次确保动作在合理范围内
            smoothed_action = np.clip(smoothed_action, -1.0, 1.0)
            
        # 将[-1, 1]范围的动作映射到实际电机转速
        rpms = self.base_rpm + smoothed_action * self.rpm_range
        
        # 保存电机历史
        self.motor_history.append(smoothed_action.copy())
        if len(self.motor_history) > self.max_history:
            self.motor_history.pop(0)
        
        # 更新上一次动作
        self.prev_action = smoothed_action
        
        # 执行一步模拟
        state, reward, done, info = self.env.step(rpms)
        
        # 创建一个额外信息字典，存储电机历史信息和当前动作
        info_dict = info.copy() if isinstance(info, dict) else {}
        info_dict.update({
            'motor_history': self.motor_history.copy(),
            'current_action': smoothed_action
        })
        
        # 返回符合gym接口的结果，添加截断标志（truncated）
        return state, reward, done, False, info_dict
    
    def close(self):
        """关闭环境"""
        self.env.close()

def make_env(device="cpu", headless=True, verbose=False, initial_rpm=5150):
    """创建环境的工厂函数，用于并行环境"""
    def _init():
        env = HoverEnv(device=device, headless=headless, verbose=verbose)
        env.hover_rpm = initial_rpm  # 设置初始悬停转速
        return IsaacDroneEnv(env=env)
    return _init

def train_drone(total_episodes=1000, num_envs=1, with_window=False, initial_rpm=5150):
    """训练无人机使用TD3算法
    
    参数:
        total_episodes: 总训练回合数
        num_envs: 并行环境数量，如设为1则不使用并行
        with_window: 是否显示可视化窗口，开启会降低训练速度
        initial_rpm: 初始电机转速，影响起始悬停状态
    """
    # 创建保存模型的目录
    log_dir = "TD3/logs/td3_drone"
    os.makedirs(log_dir, exist_ok=True)
    model_dir = "TD3/models/td3_drone"
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建环境
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 训练时关闭详细输出以加快速度，但如果开启窗口则显示信息
    verbose = with_window
    
    # 设置headless参数，根据with_window决定是否显示窗口
    headless = not with_window
    
    if num_envs > 1:
        if with_window:
            print("警告: 并行环境下无法显示多个可视化窗口，只有第一个环境会显示")
        print(f"创建{num_envs}个并行环境以加速训练...")
        env_fns = [make_env(device=device, headless=(i > 0 or headless), verbose=(i == 0 and verbose), 
                           initial_rpm=initial_rpm) 
                   for i in range(num_envs)]
        env = SubprocVecEnv(env_fns)
    else:
        # 使用单个环境
        env_hover = HoverEnv(device=device, headless=headless, verbose=verbose)
        env_hover.hover_rpm = initial_rpm  # 设置初始悬停转速
        env = IsaacDroneEnv(env=env_hover)
    
    # 设置动作噪声以促进探索
    n_actions = 4  # 四个电机
    
    # 设置适当的噪声水平
    noise_sigma = 0.1  # 探索噪声大小
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=noise_sigma * np.ones(n_actions)
    )
    
    # 设置合适的学习参数
    learning_rate = 1e-4  # 学习率
    batch_size = 256     # 批量大小
    buffer_size = 20000  # 回放缓冲区大小
    learning_starts = 1000  # 初始探索步数
    
    # 创建TD3模型
    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=0.99,  # 折扣因子
        tau=0.001,   # 目标网络软更新系数
        policy_delay=3,  # 延迟策略更新
        target_policy_noise=0.1,  # 目标动作噪声
        target_noise_clip=0.3,  # 目标噪声裁剪
        verbose=0,  # 改为0，我们将使用自定义输出
        tensorboard_log=log_dir,
        device=device
    )
    
    # 创建进度条和覆盖式输出管理
    from tqdm.auto import tqdm
    import time
    import sys
    
    # 开始训练
    print(f"开始训练TD3算法，用于无人机悬停...(总回合数: {total_episodes})")
    
    # 进度和统计数据
    start_time = time.time()
    episodes_completed = 0
    update_count = 0
    last_critic_loss = 0
    last_actor_loss = 0
    last_fps = 0
    total_steps = 0
    
    # 创建进度条
    pbar = tqdm(total=total_episodes, desc="训练进度")
    
    try:
        # 自定义回调函数，用于更新统计信息和检查回合数
        def custom_callback(_locals, _globals):
            nonlocal episodes_completed, update_count, last_critic_loss, last_actor_loss, last_fps, total_steps
            
            # 获取模型内部状态
            self_model = _locals['self']
            timesteps = self_model.num_timesteps
            
            # 更新总步数
            steps_taken = timesteps - total_steps
            total_steps = timesteps
            
            # 获取当前回合数
            current_episodes = self_model._episode_num
            
            # 检查回合数是否增加
            if current_episodes > episodes_completed:
                # 更新进度条
                episodes_added = current_episodes - episodes_completed
                pbar.update(episodes_added)
                episodes_completed = current_episodes
            
            # 获取loss信息
            if hasattr(self_model, "logger") and hasattr(self_model.logger, "name_to_value"):
                for k, v in self_model.logger.name_to_value.items():
                    if "critic_loss" in k:
                        last_critic_loss = v
                    if "actor_loss" in k:
                        last_actor_loss = v
            
            # 计算当前信息
            fps = int(timesteps / (time.time() - start_time))
            last_fps = fps
            if hasattr(self_model, "_n_updates"):
                update_count = self_model._n_updates
            
            # 清空当前行并打印信息
            clear_line = "\r" + " " * 80 + "\r"
            
            # 创建统计输出字符串
            stats = [
                f"Episodes: {current_episodes}/{total_episodes}",
                f"Steps: {timesteps}",
                f"FPS: {fps}",
                f"Time: {int(time.time() - start_time)}s"
            ]
            
            if update_count > 0:
                stats.extend([
                    f"Updates: {update_count}",
                    f"Critic Loss: {last_critic_loss:.5f}",
                ])
                
                if hasattr(self_model, "actor_loss") and self_model.actor_loss is not None:
                    stats.append(f"Actor Loss: {last_actor_loss:.5f}")
            
            # 输出当前训练统计
            sys.stdout.write(clear_line + " | ".join(stats))
            sys.stdout.flush()
            
            # 检查是否达到目标回合数
            if current_episodes >= total_episodes:
                return False  # 停止训练
            return True
            
        # 使用自定义回调进行训练
        # 设置一个很大的total_timesteps值，因为我们将通过回调函数基于回合数停止训练
        model.learn(
            total_timesteps=int(1e9),  # 设置一个足够大的值
            callback=custom_callback
        )
        
        # 保存最终模型
        model.save(f"{model_dir}/td3_drone_final")
        print(f"\n\n训练完成，模型已保存到 {model_dir}")
        
        # 打印最终统计信息
        print("\n============ 训练统计 ============")
        print(f"总回合数: {episodes_completed}")
        print(f"总步数: {model.num_timesteps}")
        print(f"平均每回合步数: {model.num_timesteps / max(1, episodes_completed):.1f}")
        print(f"平均FPS: {last_fps}")
        print(f"总时间: {int(time.time() - start_time)}秒")
        print(f"总更新次数: {update_count}")
        if last_critic_loss > 0:
            print(f"最终Critic Loss: {last_critic_loss:.5f}")
        if last_actor_loss != 0:
            print(f"最终Actor Loss: {last_actor_loss:.5f}")
        print("==================================")
        
    except KeyboardInterrupt:
        # 用户中断训练，保存当前模型
        print("\n\n训练被用户中断，保存当前模型...")
        model.save(f"{model_dir}/td3_drone_interrupted")
        print(f"中断点模型已保存到 {model_dir}/td3_drone_interrupted")
    finally:
        # 确保环境被关闭
        pbar.close()
        env.close()

def test_drone(model_path=None, initial_rpm=5150):
    """测试训练好的无人机模型，默认无限循环自动测试
    
    参数:
        model_path: 模型文件路径，为None时使用默认模型
        initial_rpm: 初始电机转速，影响起始悬停状态
    """
    # 创建环境
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 测试时保留可视化界面和详细输出
    env_hover = HoverEnv(device=device, headless=False, verbose=True)
    env_hover.hover_rpm = initial_rpm  # 设置初始悬停转速
    env = IsaacDroneEnv(env=env_hover)
    
    # 加载模型
    if model_path:
        model = TD3.load(model_path, env=env)
        print(f"已加载模型: {model_path}")
    else:
        model_path = "TD3/models/td3_drone/td3_drone_final"
        model = TD3.load(model_path, env=env)
        print(f"已加载模型: {model_path}")
    
    # 显示测试配置
    print(f"初始电机转速: {initial_rpm} RPM")
    
    # 测试统计
    test_count = 0
    total_reward_sum = 0
    max_reward = float('-inf')
    min_reward = float('inf')
    max_steps = 0
    total_steps = 0
    avg_rpms = np.zeros(4)  # 记录平均RPM
    
    print("开始无限循环测试，按Ctrl+C停止...")
    
    try:
        while True:  # 无限循环
            test_count += 1
            print(f"\n开始第 {test_count} 次测试")
            
            # 执行测试
            obs, info = env.reset()
            done = False
            total_reward = 0
            step_count = 0
            
            print("开始测试无人机悬停控制...")
            all_rpms = []
            
            while not done:
                # 使用确定性策略进行预测（无探索）
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step_count += 1
                
                # 选择性输出详细信息
                if step_count % 50 == 0:
                    high_z = obs[0] > 1.1
                    low_z = obs[0] < 0.9
                    status = "偏高" if high_z else "偏低" if low_z else "正常"
                    height_error = abs(obs[0] - 1.0)
                    print(f"\r步数: {step_count}, 高度: {obs[0]:.2f} ({status}), 倾角: {obs[2]:.3f}, 高度误差: {height_error:.3f}, 奖励: {reward:.2f}", end="")
                
                all_rpms.append(info.get('rpms', [0, 0, 0, 0]))
            
            # 更新统计数据
            total_reward_sum += total_reward
            avg_reward = total_reward_sum / test_count
            max_reward = max(max_reward, total_reward)
            min_reward = min(min_reward, total_reward)
            max_steps = max(max_steps, step_count)
            total_steps += step_count
            avg_steps = total_steps / test_count
            
            # 计算此次测试的平均RPM
            if all_rpms:
                test_avg_rpms = np.mean(all_rpms, axis=0)
                # 累加到总平均中
                avg_rpms = (avg_rpms * (test_count - 1) + test_avg_rpms) / test_count
                
                # 显示RPM信息
                print(f"平均RPM: [{test_avg_rpms[0]:.1f}, {test_avg_rpms[1]:.1f}, {test_avg_rpms[2]:.1f}, {test_avg_rpms[3]:.1f}]")
            
            # 自动生成测试结果报告
            success = step_count > 500  # 如果持续超过500步，视为成功
            status = "成功" if success else "失败"
            
            print(f"\n第 {test_count} 次测试{status}，总奖励: {total_reward:.2f}, 持续步数: {step_count}")
            print(f"统计信息: 平均奖励: {avg_reward:.2f}, 最高奖励: {max_reward:.2f}, 最低奖励: {min_reward:.2f}")
            print(f"步数统计: 平均步数: {avg_steps:.1f}, 最大步数: {max_steps}")
            
            # 提供悬停RPM建议
            if test_count > 0 and max_steps > 0:
                if max_steps < 200:
                    suggested_rpm = initial_rpm + 20
                    print(f"\n悬停时间较短，建议尝试较高的初始RPM值: {suggested_rpm}")
                elif avg_steps < 300:
                    suggested_rpm = initial_rpm + 10
                    print(f"\n悬停时间尚可，可以尝试略高的初始RPM值: {suggested_rpm}")
                else:
                    print(f"\n当前初始RPM值 {initial_rpm} 似乎表现良好，可继续使用")
                
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    
    env.close()
    
    # 打印最终测试统计信息
    print("\n============ 测试统计 ============")
    print(f"初始RPM: {initial_rpm}")
    print(f"总测试次数: {test_count}")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"最高奖励: {max_reward:.2f}")
    print(f"最低奖励: {min_reward:.2f}")
    print(f"平均步数: {avg_steps:.1f}")
    print(f"最大步数: {max_steps}")
    if test_count > 0:
        print(f"平均RPM: [{avg_rpms[0]:.1f}, {avg_rpms[1]:.1f}, {avg_rpms[2]:.1f}, {avg_rpms[3]:.1f}]")
    print("==================================")
    
    return avg_rpms  # 返回平均RPM值，方便进一步分析

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TD3算法训练无人机悬停")
    parser.add_argument("--test", action="store_true", help="测试模式而非训练模式")
    parser.add_argument("--model", type=str, default=None, help="测试时使用的模型路径")
    parser.add_argument("--episodes", type=int, default=1000, help="训练总回合数")
    parser.add_argument("--envs", type=int, default=1, help="并行环境数量，设为1禁用并行")
    parser.add_argument("--window", action="store_true", help="训练时显示可视化窗口")
    parser.add_argument("--initial-rpm", type=int, default=5150, help="初始旋翼转速，影响起始悬停状态")
    
    args = parser.parse_args()
    
    if args.test:
        test_drone(args.model, initial_rpm=args.initial_rpm)
    else:
        train_drone(total_episodes=args.episodes, num_envs=args.envs, with_window=args.window, 
                   initial_rpm=args.initial_rpm)

"""
使用说明:

1. 基本训练:
   python drone_TD3.py

2. 训练选项:
   - 使用无界面模式: 默认启用
   - 设置训练回合数: python TD3/drone_TD3.py --episodes 5000
   - 使用并行环境: python TD3/drone_TD3.py --envs 4
   - 训练时显示窗口: python TD3/drone_TD3.py --window
   - 设置初始旋翼转速: python TD3/drone_TD3.py --initial-rpm 5150
   
   最快速训练组合示例: python TD3/drone_TD3.py --episodes 5000 --envs 5
   显示窗口训练示例: python TD3/drone_TD3.py --window --episodes 1000

3. 测试模型:
   python TD3/drone_TD3.py --test                       # 使用默认模型进行测试
   python TD3/drone_TD3.py --test --model 模型路径       # 使用指定模型进行测试
   python TD3/drone_TD3.py --test --initial-rpm 5200    # 使用指定初始转速测试
   
   测试过程会自动无限循环，按Ctrl+C可随时中断并查看统计结果

4. 训练特点:
   - 基于回合数训练：不再设置总步数限制，而是设置总回合数，更符合训练逻辑
   - 宽松的活动空间：允许无人机在较大范围内探索，倾角最大30度，高度范围0.3米到1.8米
   - 扩大的电机控制范围：电机RPM范围4900-5400，动作调整幅度增大
   - 连续悬停：当无人机稳定悬停100步后，会自动结束当前回合
   - 宽松的稳定条件：高度误差<0.15米，垂直速度<0.08米/秒，倾角<0.15弧度
   
5. 为后续风力干扰研究做准备:
   - 首先找到合适的初始RPM: python drone_TD3.py --test --initial-rpm <不同值>
   - 用找到的RPM值训练模型: python drone_TD3.py --initial-rpm <找到的值>
   - 训练足够回合后测试: python drone_TD3.py --test
""" 