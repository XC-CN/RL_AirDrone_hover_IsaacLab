import numpy as np
import matplotlib.pyplot as plt
import os

def get_absolute_path(relative_path):
    """获取绝对路径"""
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建绝对路径（在当前目录下）
    return os.path.join(current_dir, relative_path)

def smooth_data(data, window_size=10, method='moving_average'):
    """
    平滑数据
    
    参数:
        data: 要平滑的数据数组
        window_size: 平滑窗口大小
        method: 平滑方法，可选 'moving_average' 或 'exponential'
    
    返回:
        平滑后的数据
    """
    if len(data) < window_size:
        return data
    
    if method == 'moving_average':
        # 使用移动平均平滑
        smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        # 处理边界情况
        pad_size = (len(data) - len(smoothed)) // 2
        if pad_size > 0:
            smoothed = np.pad(smoothed, (pad_size, len(data) - len(smoothed) - pad_size), 
                             mode='edge')
        return smoothed
    elif method == 'exponential':
        # 使用指数平滑
        alpha = 2.0 / (window_size + 1)
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    else:
        return data

def read_training_data(data_dir="outputs/plots"):
    """读取训练数据"""
    data = {}
    data_dir = get_absolute_path(data_dir)
    
    print(f"正在从 {data_dir} 读取数据...")
    
    # 读取训练奖励数据
    rewards_file = os.path.join(data_dir, 'training_rewards.csv')
    if os.path.exists(rewards_file):
        try:
            rewards_data = np.loadtxt(rewards_file, delimiter=',', skiprows=1)
            # 检查数据形状
            if rewards_data.size == 0:
                print(f"警告: 奖励数据文件为空: {rewards_file}")
                data['rewards'] = np.array([])
            elif rewards_data.ndim == 0:  # 0维数组
                data['rewards'] = np.array([rewards_data])
            elif rewards_data.ndim == 1:  # 1维数组
                data['rewards'] = rewards_data
            else:  # 多维数组
                data['rewards'] = rewards_data[:, 1]  # 假设第二列是奖励值
        except Exception as e:
            print(f"读取奖励数据时出错: {e}")
            data['rewards'] = np.array([])
    else:
        print(f"未找到训练奖励数据文件: {rewards_file}")
    
    # 读取位置误差数据
    errors_file = os.path.join(data_dir, 'position_errors.csv')
    if os.path.exists(errors_file):
        try:
            errors_data = np.loadtxt(errors_file, delimiter=',', skiprows=1)
            # 检查数据形状
            if errors_data.size == 0:
                print(f"警告: 位置误差数据文件为空: {errors_file}")
                data['errors'] = np.array([])
            elif errors_data.ndim == 0:  # 0维数组
                data['errors'] = np.array([errors_data])
            elif errors_data.ndim == 1:  # 1维数组
                data['errors'] = errors_data
            else:  # 多维数组
                data['errors'] = errors_data[:, 1]  # 假设第二列是误差值
        except Exception as e:
            print(f"读取位置误差数据时出错: {e}")
            data['errors'] = np.array([])
    else:
        print(f"未找到位置误差数据文件: {errors_file}")
    
    # 读取风力适应数据
    wind_adapt_file = os.path.join(data_dir, 'wind_adaptation.csv')
    if os.path.exists(wind_adapt_file):
        try:
            wind_data = np.loadtxt(wind_adapt_file, delimiter=',', skiprows=1)
            if wind_data.size > 0 and wind_data.ndim > 1:
                data['wind_speeds'] = wind_data[:, 0]
                data['wind_errors'] = wind_data[:, 1]
                data['recovery_times'] = wind_data[:, 2]
            else:
                print(f"警告: 风力适应数据格式不正确: {wind_adapt_file}")
        except Exception as e:
            print(f"读取风力适应数据时出错: {e}")
    
    # 读取控制输入数据
    control_file = os.path.join(data_dir, 'control_inputs.csv')
    if os.path.exists(control_file):
        try:
            control_data = np.loadtxt(control_file, delimiter=',', skiprows=1)
            if control_data.size > 0:
                data['control_inputs'] = control_data
            else:
                print(f"警告: 控制输入数据文件为空: {control_file}")
        except Exception as e:
            print(f"读取控制输入数据时出错: {e}")
    else:
        print(f"未找到控制输入数据文件: {control_file}")
    
    return data

def plot_training_curves(data, save_dir="outputs/plots", smooth_window=20, smooth_method='moving_average'):
    """绘制训练曲线"""
    if 'rewards' not in data or 'errors' not in data:
        print("缺少训练奖励或位置误差数据，无法绘制训练曲线")
        return
        
    save_dir = get_absolute_path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # 绘制奖励曲线
    plt.subplot(1, 2, 1)
    # 确保rewards是数组
    rewards = np.array(data['rewards'])
    episodes = range(len(rewards))
    
    # 平滑奖励数据
    smoothed_rewards = smooth_data(rewards, window_size=smooth_window, method=smooth_method)
    
    # 绘制原始数据和平滑后的数据
    plt.plot(episodes, rewards, alpha=0.3, color='lightblue', label='原始数据')
    plt.plot(episodes, smoothed_rewards, color='blue', linewidth=2, label='平滑数据')
    
    plt.title('训练奖励曲线')
    plt.xlabel('回合 (Episode)')
    plt.ylabel('累积奖励 (Cumulative Reward)')
    plt.grid(True)
    plt.legend()
    plt.text(0.02, 0.98, '更高的奖励表示\n更好的控制性能', 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 绘制位置误差曲线
    plt.subplot(1, 2, 2)
    # 确保errors是数组
    errors = np.array(data['errors'])
    
    # 平滑误差数据
    smoothed_errors = smooth_data(errors, window_size=smooth_window, method=smooth_method)
    
    # 绘制原始数据和平滑后的数据
    plt.plot(episodes, errors, alpha=0.3, color='lightcoral', label='原始数据')
    plt.plot(episodes, smoothed_errors, color='red', linewidth=2, label='平滑数据')
    
    plt.title('位置误差曲线')
    plt.xlabel('回合 (Episode)')
    plt.ylabel('位置误差 (Position Error, m)')
    plt.grid(True)
    plt.legend()
    plt.text(0.02, 0.98, '更低的误差表示\n更高的定位精度', 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存训练曲线到: {save_path}")

def plot_wind_adaptation(data, save_dir="outputs/plots"):
    """绘制风力适应分析图"""
    save_dir = get_absolute_path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # 位置误差图
    plt.subplot(1, 2, 1)
    plt.plot(data['wind_speeds'], data['wind_errors'], 'o-')
    plt.title('位置误差与风速关系')
    plt.xlabel('风速 (Wind Speed, m/s)')
    plt.ylabel('平均位置误差 (Average Position Error, m)')
    plt.grid(True)
    plt.text(0.02, 0.98, '表示在不同风速下\n的定位精度变化', 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 恢复时间图
    plt.subplot(1, 2, 2)
    plt.plot(data['wind_speeds'], data['recovery_times'], 'o-')
    plt.title('恢复时间与风速关系')
    plt.xlabel('风速 (Wind Speed, m/s)')
    plt.ylabel('恢复时间 (Recovery Time, s)')
    plt.grid(True)
    plt.text(0.02, 0.98, '表示从扰动恢复到\n稳定状态所需时间', 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'wind_adaptation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存风力适应分析图到: {save_path}")

def plot_control_inputs(data, save_dir="outputs/plots"):
    """绘制控制输入图"""
    save_dir = get_absolute_path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    motor_names = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(data['control_inputs'][:, i])
        plt.title(f'Motor {i+1} ({motor_names[i]}) RPM')
        plt.xlabel('Time Step')
        plt.ylabel('RPM')
        plt.ylim(4000, 6000)  # 设置y轴范围为0-10000
        plt.grid(True)
        plt.text(0.02, 0.98, f'RPM variation of Motor {i+1}\nindicates control input magnitude', 
                 transform=plt.gca().transAxes, 
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'control_inputs.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存控制输入图到: {save_path}")

if __name__ == "__main__":
    print("开始生成训练数据可视化图表...")
    
    # 读取数据
    data = read_training_data()
    
    # 绘制并保存所有图表
    if 'rewards' in data and 'errors' in data:
        # 使用移动平均平滑，窗口大小为20
        plot_training_curves(data, smooth_window=20, smooth_method='moving_average')
    else:
        print("缺少训练奖励或位置误差数据，跳过绘制训练曲线")
    
    if 'wind_speeds' in data:
        plot_wind_adaptation(data)
    else:
        print("缺少风力适应数据，跳过绘制风力适应分析图")
    
    if 'control_inputs' in data:
        plot_control_inputs(data)
    else:
        print("缺少控制输入数据，跳过绘制控制输入图")
    
    print("\n所有图表生成完成！") 