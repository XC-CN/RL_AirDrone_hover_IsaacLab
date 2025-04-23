# Reinforcement Learning-Based Quadcopter Hovering Control (Isaac Sim Environment)

## Project Overview

This project implements and compares reinforcement learning algorithms for quadcopter hovering control under wind disturbances in NVIDIA's Isaac Sim physics simulation environment. The main focus is on the Proximal Policy Optimization (PPO) algorithm, with additional implementations of Twin Delayed Deep Deterministic Policy Gradient (TD3) and Deep Q-Network (DQN) for comparison.

The project demonstrates how reinforcement learning can be applied to develop robust control policies for drones operating in challenging environments without explicit system modeling.

## Key Features

- **Multiple Reinforcement Learning Algorithms**: Implementations of PPO, TD3, and DQN for comparison
- **Wind Disturbance Simulation**: Realistic wind modeling with adjustable magnitude and direction
- **High-Fidelity Physics Engine**: Built on NVIDIA's Isaac Sim, providing accurate quadcopter dynamics
- **Comprehensive Evaluation**: Tools for evaluating control performance under various wind conditions
- **Visualization**: Training curves, position error analysis, and control response metrics

## System Requirements

### Software
- Python 3.10
- PyTorch 
- NVIDIA Isaac Sim 2022.1.0 or newer
- NVIDIA IsaacLab framework
- Windows 10 & 11

### Hardware
- CUDA-capable NVIDIA GPU (recommended for faster training)
- 8GB+ GPU memory for Isaac Sim
- 16GB+ system memory

## Installation

1. **Install NVIDIA Isaac Sim 4.5.0 using pip**:
   ```bash
   # Use the conda environment file
   conda env update -f environment.yml
   conda activate isaaclab
   
   # Install PyTorch
   # CUDA 11:
   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
   # CUDA 12:
   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
   
   # Update pip
   pip install --upgrade pip
   
   # Install Isaac Sim
   pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
   ```

2. **Clone this repository and install Isaac Lab**:
   ```bash
   # Clone the IsaacLab repository in your workspace
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab
   
   # Install IsaacLab extensions and dependencies
   ./isaaclab.bat --install   # On Windows
   
   # Clone this project repository
   git clone https://github.com/yourusername/RL_AirDrone_hover_IsaacLab.git
   cd RL_AirDrone_hover_IsaacLab
   ```
  

## Verifying the installation

To verify that both Isaac Sim and this project are installed correctly:

```bash
# Test Isaac Sim installation
isaacsim

# Test IsaacLab installation
cd IsaacLab
isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py  # On Windows

# Test this project
# The model that is run here is the one that has already been trained
cd RL_AirDrone_hover_IsaacLab
python PPO/drone_PPO.py evaluate --wind --model PPO/models/best_model.pt
```

## Project Structure

```
RL_AirDrone_hover_IsaacLab/
├── PPO/
│   ├── drone_PPO.py          # Main PPO implementation
│   ├── figures/              # Training curves and results visualization
│   └── models/               # Saved PPO models
├── TD3/
│   ├── drone_TD3.py          # TD3 implementation  
│   └── models/               # Saved TD3 models
├── DQN/
│   ├── drone_DQN.py          # DQN implementation
│   └── models/               # Saved DQN models
├── Report/                   # Project report and analysis
├── environment.yml           # Conda environment definition
└── README.md                 # This file
```

## Usage

### Training

```bash
# Train using PPO algorithm (default, recommended)
python PPO/drone_PPO.py 2000

# Train with wind disturbances
python PPO/drone_PPO.py 2000 --wind

# Train with GUI for visualization (slower)
python PPO/drone_PPO.py 2000 --wind --gui

# Train using TD3 or DQN (for comparison)
python TD3/drone_TD3.py --episodes 2000 
python DQN/drone_DQN.py --episodes 2000 
```

### Evaluation

```bash
# Evaluate trained PPO model with visualization
python PPO/drone_PPO.py evaluate --model PPO/models/quad_hover_ppo_best_model.pt

# Evaluate with wind disturbances
python PPO/drone_PPO.py evaluate --model PPO/models/quad_hover_ppo_best_model.pt --wind

# Evaluate other algorithms
python TD3/drone_TD3.py evaluate --model TD3/models/best_model.pt
python DQN/drone_DQN.py evaluate --model DQN/models/best_model.pt
```

## Algorithm Comparison

| Algorithm | Stability | Sample Efficiency | Performance in Wind | Convergence Speed |
|-----------|-----------|-------------------|---------------------|-------------------|
| PPO       | Excellent | Good              | Excellent           | Medium            |
| TD3       | Poor      | Medium            | Poor                | Slow              |
| DQN       | Fair      | Poor              | Poor                | Fast              |

Our results show that PPO significantly outperforms both TD3 and DQN for this continuous control task. PPO achieves stable hovering even under varying wind conditions, while the other algorithms struggle with basic stability.

## Key Results

- PPO successfully learns to stabilize the quadcopter at a target hover position (1m height)
- The trained policy can maintain position with average error <0.1m in no-wind conditions
- Under wind disturbances up to 0.5N, position error remains below 0.2m
- The controller adapts to changing wind directions by automatically adjusting attitude angles

## Acknowledgments

This project uses NVIDIA's Isaac Sim and IsaacLab framework. The RL algorithms are implemented in PyTorch and draw inspiration from the following papers:

- PPO: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
- TD3: Fujimoto, S., Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. ICML.
- DQN: Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. Nature.
