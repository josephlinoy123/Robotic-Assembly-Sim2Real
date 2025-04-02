import os
import numpy as np
from typing import List, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from .environments import RobotEnv
import matplotlib.pyplot as plt

def load_generated_data(data_path: str) -> List[np.ndarray]:
    """Load generated synthetic data for evaluation"""
    synthetic_data = []
    for task in range(1, 11):
        file_path = os.path.join(data_path, f"Task{task:02d}_synthetic.npy")
        synthetic_data.append(np.load(file_path))
    return synthetic_data

def evaluate_agent(
    model_path: str,
    data_path: str,
    output_dir: str,
    n_episodes: int = 10,
    n_envs: int = 4
) -> Tuple[float, float, List[np.ndarray], List[np.ndarray]]:
    """Evaluate RL agent and generate metrics"""
    # Load model and data
    model = PPO.load(model_path)
    synthetic_data = load_generated_data(data_path)
    
    # Create evaluation environment
    env = make_vec_env(lambda: RobotEnv(synthetic_data), n_envs=n_envs)
    
    # Evaluation metrics storage
    all_rewards = []
    all_actions = []
    all_states = []

    for episode in range(n_episodes):
        obs = env.reset()
        episode_rewards = []
        episode_actions = []
        episode_states = []
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            
            episode_rewards.append(reward)
            episode_actions.append(action)
            episode_states.append(obs)
            
        all_rewards.append(np.mean(episode_rewards))
        all_actions.append(np.concatenate(episode_actions))
        all_states.append(np.concatenate(episode_states))
    
    return (
        np.mean(all_rewards),
        np.std(all_rewards),
        all_actions,
        all_states
    )

def plot_behavior(
    actions: List[np.ndarray],
    states: List[np.ndarray],
    output_path: str,
    features: List[str] = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
):
    """Generate and save behavior visualization plots"""
    for ep_idx, (ep_actions, ep_states) in enumerate(zip(actions, states)):
        plt.figure(figsize=(15, 8))
        plt.suptitle(f"Agent Behavior - Episode {ep_idx+1}")
        
        for i in range(6):
            plt.subplot(2, 3, i+1)
            plt.plot(ep_states[:, i], label='Reference')
            plt.plot(ep_actions[:, i], label='Agent', linestyle='--')
            plt.title(features[i])
            plt.legend()
        
        plt.tight_layout()
        plot_file = os.path.join(output_path, f"behavior_episode_{ep_idx+1}.png")
        plt.savefig(plot_file)
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate RL agent')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to generated data directory')
    parser.add_argument('--output', type=str, default="results",
                      help='Output directory for plots and metrics')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of evaluation episodes')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    mean_reward, std_reward, actions, states = evaluate_agent(
        args.model,
        args.data,
        args.output,
        args.episodes
    )
    
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    plot_behavior(actions, states, args.output)