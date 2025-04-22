import os
import sys
import torch
import numpy as np
import time
import csv
from datetime import datetime
import argparse

# 导入项目中的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drone_PPO import HoverEnv, PPOActor

class ModelTester:
    def __init__(self, model_path, enable_gui=False, enable_wind=True, min_wind_strength=0.0, max_wind_strength=0.5):
        self.model_path = model_path
        self.enable_gui = enable_gui
        self.enable_wind = enable_wind
        self.min_wind_strength = min_wind_strength
        self.max_wind_strength = max_wind_strength
        
        # 设置参数
        self.state_dim = 18
        self.action_dim = 4
        self.min_rpm = 4500
        self.max_rpm = 5500
        self.rpm_range = self.max_rpm - self.min_rpm
        
        # 设置设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 创建环境
        self.env = HoverEnv(device=self.device, enable_gui=self.enable_gui, enable_wind=self.enable_wind)
        
        # 修改环境中的风力设置
        if self.enable_wind:
            self.env.min_wind_strength = self.min_wind_strength
            self.env.max_wind_strength = self.max_wind_strength
            print(f"已设置风力范围: {self.min_wind_strength}-{self.max_wind_strength} 牛顿")
        
        # 加载模型
        self.actor = PPOActor(self.state_dim, self.action_dim).to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor.eval()
        
        print(f"模型已加载: {self.model_path}")
        
        # 获取当前脚本的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 创建CSV文件
        self.results_dir = os.path.join(current_dir, "data", "test_results")
        os.makedirs(self.results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(self.results_dir, f"model_test_results_{timestamp}.csv")
        
        # 初始化CSV文件和标题
        with open(self.csv_filename, 'w', newline='') as csvfile:
            fieldnames = [
                'trial_num', 
                'wind_strength', 
                'wind_direction', 
                'hover_time', 
                'stabilization_time', 
                'position_error_x', 
                'position_error_y', 
                'position_error_z', 
                'position_error_total',
                'max_position_error',
                'average_velocity_error',
                'max_velocity_error',
                'average_attitude_error',
                'max_attitude_error',
                'success'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        print(f"结果将保存到: {self.csv_filename}")
    
    def run_single_test(self, trial_num):
        """运行单次测试并返回结果"""
        state = self.env.reset()
        
        # 获取初始风力设置
        wind_strength = self.env.wind_strength if self.enable_wind else 0.0
        wind_direction = self.env.wind_direction_deg if self.enable_wind else 0.0
        
        done = False
        steps = 0
        stable_steps = 0  # 稳定悬停步数
        stabilization_time = -1  # 恢复稳定所需时间，-1表示未恢复稳定
        
        # 记录误差数据
        position_errors = []  # 位置误差
        velocity_errors = []  # 速度误差
        attitude_errors = []  # 姿态误差
        
        # 记录当前是否处于稳定状态
        is_stable = False
        
        if self.enable_gui:
            print(f"\n=== 测试 #{trial_num} ===")
            if self.enable_wind:
                print(f"风力设置: 强度={wind_strength:.2f}牛顿, 方向={wind_direction:.1f}度")
        
        # 修改最大步数为300，超过直接结束
        max_steps = 300
        while not done and steps < max_steps:
            # 转换状态为tensor
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            
            # 使用模型预测动作（确定性模式）
            with torch.no_grad():
                action = self.actor.get_action(state_tensor, deterministic=True)
            
            # 将动作从[-1,1]范围映射到[min_rpm, max_rpm]
            actions_np = action.cpu().numpy()
            rpm_action = self.min_rpm + (actions_np + 1) * 0.5 * self.rpm_range
            rpm_action = np.clip(rpm_action, self.min_rpm, self.max_rpm)
            
            # 执行动作并获取下一个状态
            next_state, reward, done, _ = self.env.step(rpm_action)
            
            # 获取当前状态数据
            pos = next_state[0:3]  # [x, y, z]
            target_pos = next_state[3:6]  # [target_x, target_y, target_z]
            roll, pitch, yaw = next_state[6:9]  # [roll, pitch, yaw]
            lin_vel = next_state[9:12]  # [vx, vy, vz]
            ang_vel = next_state[12:15]  # [wx, wy, wz]
            position_error = next_state[15:18]  # [x_error, y_error, z_error]
            
            # 计算误差
            pos_error_total = np.linalg.norm(position_error)
            vel_error_total = np.linalg.norm(lin_vel)
            att_error_total = np.sqrt(roll**2 + pitch**2)
            
            # 保存误差数据
            position_errors.append(position_error)
            velocity_errors.append(vel_error_total)
            attitude_errors.append(att_error_total)
            
            # 检查无人机是否稳定悬停
            lin_vel_stable = np.all(np.abs(lin_vel) < 0.05)  # velocity_threshold = 0.05 m/s
            ang_vel_stable = np.all(np.abs(ang_vel) < 0.1)   # angular_velocity_threshold = 0.1 rad/s
            attitude_stable = (abs(roll) < 0.05 and          # attitude_threshold = 0.05 rad
                              abs(pitch) < 0.05)
            position_stable = pos_error_total < 0.1          # position_threshold = 0.1 m
            
            current_is_stable = lin_vel_stable and ang_vel_stable and attitude_stable and position_stable
            
            # 如果无人机刚刚恢复稳定，记录恢复时间
            if current_is_stable and not is_stable and steps > 50:  # 忽略前50步的稳定状态判断
                if stabilization_time == -1:  # 只记录第一次恢复稳定的时间
                    stabilization_time = steps * self.env.dt
            
            # 更新稳定状态
            is_stable = current_is_stable
            
            # 如果稳定，增加稳定步数
            if is_stable:
                stable_steps += 1
            
            # 打印状态 (仅在GUI模式)
            if self.enable_gui and steps % 10 == 0:
                print(f"\r步骤: {steps:4d} | 位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | "
                      f"误差: {pos_error_total:.3f} | 稳定步数: {stable_steps}", end="")
            
            # 更新状态
            state = next_state
            steps += 1
        
        # 如果达到最大步数，打印超时信息
        if steps >= max_steps and self.enable_gui:
            print(f"\n测试超时: 达到最大步数 {max_steps}")
            
        # 计算悬停时间 (稳定步数 * 时间步长)
        hover_time = stable_steps * self.env.dt
        
        # 计算平均和最大误差
        position_errors = np.array(position_errors)
        avg_position_error = np.mean([np.linalg.norm(err) for err in position_errors])
        max_position_error = np.max([np.linalg.norm(err) for err in position_errors])
        
        avg_velocity_error = np.mean(velocity_errors)
        max_velocity_error = np.max(velocity_errors)
        
        avg_attitude_error = np.mean(attitude_errors)
        max_attitude_error = np.max(attitude_errors)
        
        # 判断测试是否成功 (至少100步稳定悬停)
        success = stable_steps >= 100
        
        # 获取最终位置误差
        final_position_error = position_errors[-1] if len(position_errors) > 0 else [0, 0, 0]
        
        # 记录结果
        result = {
            'trial_num': trial_num,
            'wind_strength': wind_strength,
            'wind_direction': wind_direction,
            'hover_time': hover_time,
            'stabilization_time': stabilization_time,
            'position_error_x': final_position_error[0],
            'position_error_y': final_position_error[1],
            'position_error_z': final_position_error[2],
            'position_error_total': np.linalg.norm(final_position_error),
            'max_position_error': max_position_error,
            'average_velocity_error': avg_velocity_error,
            'max_velocity_error': max_velocity_error,
            'average_attitude_error': avg_attitude_error,
            'max_attitude_error': max_attitude_error,
            'success': 1 if success else 0
        }
        
        if self.enable_gui:
            print(f"\n测试 #{trial_num} 完成")
            print(f"悬停时间: {hover_time:.2f}秒")
            print(f"恢复稳定时间: {stabilization_time:.2f}秒" if stabilization_time != -1 else "未恢复稳定")
            print(f"最终位置误差: {np.linalg.norm(final_position_error):.3f}米")
            print(f"测试结果: {'成功' if success else '失败'}")
        
        return result
    
    def run_tests(self, num_trials=100):
        """运行多次测试并保存结果到CSV文件"""
        print(f"开始进行 {num_trials} 次测试...")
        
        for i in range(num_trials):
            # 运行单次测试
            result = self.run_single_test(i + 1)
            
            # 将结果写入CSV文件
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(result.keys()))
                writer.writerow(result)
            
            # 打印进度
            print(f"完成测试 {i+1}/{num_trials} ({(i+1)/num_trials*100:.1f}%)")
        
        print(f"所有测试完成，结果已保存至 {self.csv_filename}")
    
    def close(self):
        """关闭环境"""
        self.env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试训练好的PPO模型")
    parser.add_argument("--model", type=str, default="D:/Work/IsaacLab/Project/models/ppo/best_model.pt", help="模型路径")
    parser.add_argument("--gui", action="store_true", help="启用GUI模式")
    parser.add_argument("--no-wind", action="store_true", help="禁用风力干扰")
    parser.add_argument("--min-wind", type=float, default=0.0, help="最小风力强度(牛顿)")
    parser.add_argument("--max-wind", type=float, default=0.5, help="最大风力强度(牛顿)")
    parser.add_argument("--trials", type=int, default=100, help="测试次数")
    
    args = parser.parse_args()
    
    # 打印参数
    print("测试参数:")
    print(f"模型路径: {args.model}")
    print(f"GUI模式: {'启用' if args.gui else '禁用'}")
    print(f"风力干扰: {'禁用' if args.no_wind else '启用'}")
    print(f"风力范围: {args.min_wind}-{args.max_wind} 牛顿")
    print(f"测试次数: {args.trials}")
    
    try:
        # 创建测试器
        tester = ModelTester(
            model_path=args.model,
            enable_gui=args.gui,
            enable_wind=not args.no_wind,
            min_wind_strength=args.min_wind,
            max_wind_strength=args.max_wind
        )
        
        # 运行测试
        tester.run_tests(num_trials=args.trials)
    except Exception as e:
        print(f"测试过程中出错: {e}")
    finally:
        # 确保环境关闭
        if 'tester' in locals():
            tester.close() 