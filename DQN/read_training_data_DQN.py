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

def read_training_data(data_dir="data"):
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
    
    return data

def plot_training_curves(data, save_dir="data", smooth_window=20, smooth_method='moving_average'):
    """绘制训练曲线 - 将奖励和误差曲线分别显示在左右两个子图中"""
    if 'rewards' not in data or 'errors' not in data:
        print("缺少训练奖励或位置误差数据，无法绘制训练曲线")
        return
        
    save_dir = get_absolute_path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建1x2的子图布局（一行两列）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 确保数据是数组
    rewards = np.array(data['rewards'])
    errors = np.array(data['errors'])
    episodes = range(len(rewards))
    
    # 平滑数据
    smoothed_rewards = smooth_data(rewards, window_size=smooth_window, method=smooth_method)
    smoothed_errors = smooth_data(errors, window_size=smooth_window, method=smooth_method)
    
    # 为整个图表设置一个总标题
    fig.suptitle('DQN Training Performance', fontsize=16, y=0.98)
    
    # 左侧子图 - 奖励
    color1 = 'blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cumulative Reward')
    ax1.plot(episodes, rewards, alpha=0.3, color='lightblue', label='Original Data')
    ax1.plot(episodes, smoothed_rewards, color=color1, linewidth=2, label='Smoothed Data')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Training Rewards')
    
    # 添加说明文本到左侧子图
    ax1.text(0.05, 0.95, 'Higher rewards indicate\nbetter control performance', 
             transform=ax1.transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加图例到左侧子图
    ax1.legend(loc='upper right')
    
    # 右侧子图 - 误差
    color2 = 'red'
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Position Error (m)')
    ax2.plot(episodes, errors, alpha=0.3, color='lightcoral', label='Original Data')
    ax2.plot(episodes, smoothed_errors, color=color2, linewidth=2, label='Smoothed Data')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Position Errors')
    
    # 设置右侧子图的显示范围，确保从0开始
    error_max = max(errors) * 1.1
    ax2.set_ylim(0, error_max)
    
    # 添加说明文本到右侧子图
    ax2.text(0.05, 0.95, 'Lower errors indicate\nhigher positioning accuracy', 
             transform=ax2.transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加图例到右侧子图
    ax2.legend(loc='upper right')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(save_dir, 'training_curves_dqn.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存DQN训练曲线到: {save_path}")

if __name__ == "__main__":
    print("开始生成DQN训练数据可视化图表...")
    
    # 读取数据
    data = read_training_data()
    
    # 绘制并保存图表
    if 'rewards' in data and 'errors' in data:
        # 使用移动平均平滑，窗口大小为20
        plot_training_curves(data, smooth_window=20, smooth_method='moving_average')
    else:
        print("缺少训练奖励或位置误差数据，无法绘制训练曲线")
    
    print("\n所有图表生成完成！") 