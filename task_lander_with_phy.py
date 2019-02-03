import numpy as np
import math
from physics_sim import PhysicsSim
import gym.envs.box2d.lunar_lander as lunder


class TaskLanderWithPhy():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, env=None ,init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 4

        self.state_size = self.action_repeat  * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 40

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
        
        #-------------------------------------------------------------

        #-------------------------------------------------------------
        
        self.env= env
        self.env.continuous = False
        
#         self.action_repeat = 3
#         self.state_size = self.action_repeat * np.prod(env.observation_space.shape)
        
#         self.action_size = np.prod(env.action_space.shape)
        #-------------------------------------------------------------
        # lunder.demo_heuristic_lander(self.env, render=True)


#     def step(self, action):
#         """Uses action to obtain next state, reward, done."""
#         total_reward = 0
#         reward = 0
#         done = 0
#         obs = []
#         for _ in range(self.action_repeat):
#             ob, reward, done, info = self.env.step(action)
#             done = self.sim.next_timestep(action)
            
#             total_reward += reward
#             obs.append(ob)
        
#         next_state = np.concatenate(obs)
        
#         return next_state, reward, done
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.3*(abs ( self.sim.v[2] ).sum() )
        # print('velocity is ' , self.sim.v[2] , 'position',self.sim.pose[:3] ,'target' ,self.target_pos )
        # print ('reward',reward)
        # if self.env.game_over or abs(self.env.state[0]) >= 1.0:
        #     done   = True
        #     reward = -100
        # if not self.env.lander.awake:
        #     done   = True
        #     reward = +100
        # print(self.sim.pose[:3])
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        # total_reward - 0
        pose_all = []
        for _ in range(self.action_repeat):
            # ob, reward, done, info = self.env.step(action)
            # total_reward  += reward
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            # self.sim.pose = ob
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done
        
    def reset(self):
        """Reset the sim to start a new episode."""
        ob = self.env.reset()
        state = np.concatenate([ob] * self.action_repeat)
        return state   
        