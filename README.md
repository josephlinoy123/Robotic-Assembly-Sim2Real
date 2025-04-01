# Bridging the Sim-to-Real Gap in Robotic Assembly Using Generative AI and RL

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)

This repository contains the source code and related materials for my master's thesis:  
**"Leveraging Generative AI and Reinforcement Learning to Improve Robot-based Assembly Task Simulations"**  
*Deggendorf Institute of Technology, 2025*

![Image](https://github.com/user-attachments/assets/9a507da4-00f4-4f01-872d-f93cb6ee46fc)

## 📝 Abstract
This repository contains the complete implementation of a novel two-stage approach combining:
1. **VAE-LSTM models** for synthetic force/torque data generation
2. **Proximal Policy Optimization (PPO)** for RL-based control policy training

Key achievements:
- **98.98% MSE improvement** in synthetic force data realism vs physics-based simulation
- Successful sim-to-real transfer learning for robotic assembly tasks
- Custom `RobotEnv` Gym environment for RL training

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended)

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/Robotic-Assembly-Sim2Real.git
cd Robotic-Assembly-Sim2Real
pip install -r requirements.txt
```

### Dataset Preparation
1. Download the [CSIRO Manipulation Benchmark Dataset]([https://example.com/dataset](https://research.csiro.au/robotics/manipulation-benchmark/))
2. Place raw data in `data/raw/`

## 🧠 Model Architecture
### Two-Stage Methodology
```mermaid
graph LR
A[Real+Simulated Data] --> B[VAE-LSTM]
B --> C[Synthetic FT Data]
C --> D[PPO Agent]
D --> E[Robotic Controller]
```

## 💻 Usage
### 1. Generate Synthetic Data
```bash
# Train ForceVAE model
python src/vae_lstm/train.py --config configs/force_vae.yaml

# Generate synthetic trajectories
python src/vae_lstm/generate_data.py --task 01 --output data/generated/
```

### 2. Train RL Agent
```bash
python src/rl_training/train_ppo.py \
  --env RobotEnv-v1 \
  --total_timesteps 1000000 \
  --log_dir logs/
```

## 📂 Repository Structure
```
.
├── data/                   # Dataset directories
├── docs/                   # Thesis documentation
├── models/                 # Pretrained models
├── src/                    # Source code
│   ├── data_processing/    # Data pipelines
│   ├── vae_lstm/           # Generative models
│   └── rl_training/        # RL implementations
├── experiments/            # Jupyter notebooks
├── requirements.txt        # Python dependencies
└── LICENSE
```

## 📚 Citation
If you use this work in your research, please cite:
```bibtex
@mastersthesis{Arakkal2025Sim2Real,
  author  = {Joseph Linoy Arakkal},
  title   = {Leveraging Generative AI and Reinforcement Learning to Improve Robot-based Assembly Task Simulations},
  school  = {Deggendorf Institute of Technology},
  year    = {2025},
  url     = {https://github.com/YOUR_USERNAME/Robotic-Assembly-Sim2Real}
}
```

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Supervisors: Mr. Ginu Paul Alunkal and Dr. Alper Yaman
- Dataset providers: CSIRO Robotics
