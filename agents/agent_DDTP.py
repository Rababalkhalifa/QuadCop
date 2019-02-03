from agents.actor import Actor
from agents.critic import Critic
from agents.replay_buffer import ReplayBuffer
from agents.ou_noise import OUNoise
from keras import layers, models, optimizers
import numpy as np
import gym.envs.box2d.lunar_lander as lunder

import random


class DDPG_Land():
    def __init__(self, task, seed=None, render=False):
        
        
        self.env = task.env
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
  

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        
        
        
        self.total_reward = 0
        self.steps = 0
        self.action_repeat = 3
        self.render = render
        
        # Score tracker and learning parameters
        self.score = -np.inf
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1
        
        #counter
        self.count = 0

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        
        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(1, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        
        
        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.01  # for soft update of target parameters
        
    def act(self,s):
#         # print('act')
#         # a = lunder.heuristic(self.env, s)
#         # 1. Testing. 
#         # 2. Demonstration rollout.
#         angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
#         if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
#         if angle_targ < -0.4: angle_targ = -0.4
#         hover_targ = 0.55*np.abs(s[0])           # target y should be proporional to horizontal offset

#         # PID controller: s[4] angle, s[5] angularSpeed
#         angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
#         #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

#         # PID controller: s[1] vertical coordinate s[3] vertical speed
#         hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
#         #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

#         if s[6] or s[7]: # legs have contact
#             angle_todo = 0
#             hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

#         if self.env.continuous:
#             a = np.array( [hover_todo*20 - 1, -angle_todo*20] )
#             a = np.clip(a, -1, +1)
#         else:
#             a = 0
#             if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
#             elif angle_todo < -0.05: a = 3
#             elif angle_todo > +0.05: a = 1
#         # return a
        # state = s
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(s, [-1, 24])
        action = self.actor_local.model.predict(state)[0]
        
        return list(action + self.noise.sample())

    def step(self, action, reward, next_state, done):
        # print ("step")
        # ob, reward, done, info = self.env.step(action)
        # print(ob)
        # next_state = ob
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state , done)
        
        self.count += 1
        self.total_reward += reward
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state
        
        #from the tutorial SRC
        self.score += reward
        
        if done:
            if self.score > self.best_score:
                self.best_score = self.score
        
    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
        
#         #from the tutorial SRC
#         self.score += reward
        
        
#         if done:
            
#             if self.score > self.best_score:
#                 self.best_score = self.score
                
#         # return ob, reward, done

                
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)  
        
#         # from policy search
        
#         # Learn by random policy search, using a reward-based score
#         # self.score = self.total_reward / float(self.count) if self.count else 0.0
#         # if self.score > self.best_score:
#         #     self.best_score = self.score
#         #     self.best_w = self.w
#         #     self.noise_scale = max(0.5 * self.noise_scale, 0.01)
#         # else:
#         #     self.w = self.best_w
#         #     self.noise_scale = min(2.0 * self.noise_scale, 3.2)
#         # self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
        
    def reset(self):
        self.steps = 0
        self.total_reward = 0
        self.count = 0
        self.score = 0
        
        
        """Reset the sim to start a new episode."""
        ob = self.env.reset()
        state = np.concatenate([ob] * self.action_repeat)

        self.last_state = state
        return state 