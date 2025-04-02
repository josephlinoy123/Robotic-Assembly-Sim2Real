import gym
import numpy as np
import random
from gym import spaces

class RobotEnv(gym.Env):
    """Custom environment for robotic assembly tasks"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, synthetic_data):
        super().__init__()
        self.data = synthetic_data
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,))  # Force + Torque
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        self.task_idx = 0
        self.step_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        self.task_idx = np.random.randint(0, len(self.data))
        self.step_idx = 0
        return self.data[self.task_idx][self.step_idx], {}

    def step(self, action):
        self.step_idx += 1
        done = self.step_idx >= len(self.data[self.task_idx]) - 1
        next_state = self.data[self.task_idx][self.step_idx]
        reward = -np.linalg.norm(action - next_state)
        return next_state, reward, done, False, {}

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Task: {self.task_idx+1}, Step: {self.step_idx+1}")