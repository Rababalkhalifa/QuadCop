import numpy as np
import math

import gym.envs.box2d.lunar_lander as lunder


class TaskLander():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, env):
        
        self.env= env
        self.env.continuous = False
        
        self.action_repeat = 3
        self.state_size = self.action_repeat * np.prod(env.observation_space.shape)
        
        self.action_low = 10
        self.action_high = 900
        self.action_size = np.prod(env.action_space.shape)
        
        # lunder.demo_heuristic_lander(self.env, render=True)

    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        total_reward = 0
        reward = 0
        done = 0
        obs = []
        for _ in range(self.action_repeat):
            ob, reward, done, info = self.env.step(action)
            total_reward += reward
            obs.append(ob)

        next_state = np.concatenate(obs)
        
                
        return next_state, reward, done
        
    def reset(self):
        """Reset the sim to start a new episode."""
        ob = self.env.reset()
        state = np.concatenate([ob] * self.action_repeat)
        return state   
        