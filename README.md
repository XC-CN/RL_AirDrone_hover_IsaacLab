# 四旋翼无人机强化学习控制系统

## 项目简介

本项目使用IsaacLab物理仿真环境，结合多种强化学习算法（PPO、DQN、TD3）实现了四旋翼无人机的悬停和姿态控制。项目模拟了真实环境中的风力干扰，提高了控制系统的鲁棒性和稳定性。

## 主要功能

- **多种强化学习算法**：实现了PPO（近端策略优化）、DQN（深度Q网络）和TD3（双延迟深度确定策略梯度）算法
- **风力干扰模拟**：支持随机风力干扰，模拟真实飞行环境
- **精确姿态控制**：实现四旋翼无人机的稳定悬停和姿态控制
- **仿真可视化**：支持GUI模式直观观察无人机飞行状态
- **性能评估**：提供模型评估工具，计算平均奖励、飞行时间和悬停成功率

## 环境要求

- Python 3.6+
- PyTorch 1.9+
- NVIDIA Isaac Sim
- CUDA支持的NVIDIA GPU（推荐）

## 安装步骤

1. 安装NVIDIA Isaac Sim：[官方安装指南](https://developer.nvidia.com/isaac-sim)

2. 安装项目依赖：
```bash
pip install torch numpy matplotlib
```

3. 克隆项目仓库：
```bash
git clone https://github.com/your-username/IsaacLab.git
cd IsaacLab
```

## 使用方法

### 训练模型

```bash
# 使用PPO算法训练无人机悬停（无风力干扰）
python Project/drone_PPO.py 1000

# 启用风力干扰进行训练
python Project/drone_PPO.py 2000 --wind

# 使用GUI模式进行训练（速度较慢但可视化效果好）
python Project/drone_PPO.py 1000 --gui

# 使用DQN或TD3算法训练
python Project/drone_DQN.py 1000
python Project/drone_TD3.py 1000
```

### 评估模型

```bash
# 评估训练好的PPO模型（默认使用最佳模型）
python Project/drone_PPO.py evaluate

# 评估特定模型在有风力干扰环境中的性能
python Project/drone_PPO.py evaluate --model models/ppo/best_model.pt --wind

# 评估DQN或TD3模型
python Project/drone_DQN.py evaluate
python Project/drone_TD3.py evaluate
```

## 项目结构

- `Project/`：主项目目录
  - `drone_PPO.py`：PPO算法实现与无人机控制
  - `drone_DQN.py`：DQN算法实现
  - `drone_TD3.py`：TD3算法实现
  - `quadcopter.py`：四旋翼无人机物理模型
  - `test_crazyflie_attitude.py`：姿态控制测试脚本
  - `models/`：预训练模型存储目录
  - `data/`：训练数据存储目录
  - `figures/`：图表和可视化结果
  - `Report/`：项目报告和文档

## 算法比较

| 算法 | 稳定性 | 训练速度 | 样本效率 | 适用场景 |
|-----|-------|---------|---------|---------|
| PPO | 高 | 中 | 高 | 连续动作空间，需要稳定性 |
| DQN | 中 | 快 | 低 | 离散动作空间，探索性强 |
| TD3 | 高 | 慢 | 高 | 连续动作空间，高精度控制 |

## 引用

如果您使用本项目进行研究，请引用：

```bibtex
@misc{DroneRL2023,
  author = {Wu Zining},
  title = {四旋翼无人机强化学习控制系统},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/your-username/IsaacLab}}
}
```
