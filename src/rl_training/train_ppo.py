import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from .environments import RobotEnv

def train_rl(generated_data_path: str, 
             output_dir: str = "rl_models",
             total_timesteps: int = 1_000_000,
             learning_rate: float = 3e-4,
             n_envs: int = 4):
    """
    Train PPO agent on generated synthetic data
    """
    # Load generated data
    synthetic_data = []
    for task in range(1, 11):
        data = np.load(os.path.join(generated_data_path, f"Task{task:02d}_synthetic.npy"))
        synthetic_data.append(data)
    
    # Create vectorized environment
    env = make_vec_env(
        lambda: RobotEnv(synthetic_data),
        n_envs=n_envs
    )
    
    # Initialize and train PPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "tensorboard")
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, "ppo_robot"))
    print(f"Saved trained model to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train RL agent on synthetic data')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to directory with generated .npy files')
    parser.add_argument('--output', type=str, default="rl_models",
                      help='Output directory for trained model')
    parser.add_argument('--timesteps', type=int, default=1000000,
                      help='Total training timesteps')
    parser.add_argument('--lr', type=float, default=3e-4,
                      help='Learning rate')
    args = parser.parse_args()
    
    train_rl(
        generated_data_path=args.data,
        output_dir=args.output,
        total_timesteps=args.timesteps,
        learning_rate=args.lr
    )