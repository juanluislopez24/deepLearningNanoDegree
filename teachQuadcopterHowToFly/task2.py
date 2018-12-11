import numpy as np
from physics_sim import PhysicsSim
from numpy import linalg as LA
import math

def sigmoid(x):
    return (2 / (1 + np.exp(-x)))-1


def sigmoid2(x):
    return (1 / (1 + np.exp(-x)))

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
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
        self.action_repeat = 1
        
        self.init_pose = init_pose
        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
        #print("SAFSFAS")
        #print(self.target_pos - self.init_pose[:3])
        
        self.pose_normalizer = LA.norm(self.target_pos - self.init_pose[:3])
        print("SAFSFAS")
        print(self.pose_normalizer)

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #task.sim.pose (the position of the quadcopter in (x,y,z) dimensions and the Euler angles),
        #task.sim.v (the velocity of the quadcopter in (x,y,z) dimensions), and
        #task.sim.angular_v (radians/second for each of the three Euler angles).
        
        #velocity same direction as target position a dot b = magnitude(a)*magnitude(b)
        
        #if same direction no penalty since dot product between a and b = norm(a)*norm(b)
        #penalty2 = sigmoid(abs(np.dot(self.sim.v, (self.target_pos-self.sim.pose[:3]))-(LA.norm(self.sim.v)*LA.norm(self.target_pos-self.sim.pose[:3]))))
        
        #since the formula gives 0 if the vectors are pointing the same direction,when applying the sigmoid we need to subtract 0.5 so we dont punish a good direction since sigmoid of 0 = 0.5
        direction_penalty = sigmoid(abs(np.dot(self.sim.v/LA.norm(self.sim.v), (self.target_pos-self.sim.pose[:3])/LA.norm(self.target_pos-self.sim.pose[:3]))-(LA.norm(self.sim.v/LA.norm(self.sim.v))*LA.norm((self.target_pos-self.sim.pose[:3])/LA.norm(self.target_pos-self.sim.pose[:3])))))
        
        direction_penalty2 = sigmoid2(abs(np.dot(self.sim.v/LA.norm(self.sim.v), (self.target_pos-self.sim.pose[:3])/LA.norm(self.target_pos-self.sim.pose[:3]))-(LA.norm(self.sim.v/LA.norm(self.sim.v))*LA.norm((self.target_pos-self.sim.pose[:3])/LA.norm(self.target_pos-self.sim.pose[:3]))))) - 0.5
        
        
        angle = math.acos(np.dot(self.sim.v, (self.target_pos-self.sim.pose[:3]))/(LA.norm(self.sim.v)*LA.norm(self.target_pos-self.sim.pose[:3])))
        #print (angle)
        
        distance = LA.norm(self.sim.pose[:3] - self.target_pos)
        
        distance_penalty = distance / self.pose_normalizer
        
        reward_distance = 1/(1+distance)
        
        position_penalty = 0.5*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        
        reward = 1.60 - (angle * distance_penalty) + (1 - distance_penalty) + (2/distance)
        
        if(distance_penalty<0):
            while True:
                print("SFAFSAFAFSA")
        
        reward2 = 1. - .85*sigmoid2((abs(self.sim.pose[:3]-self.target_pos)).sum()) - .3*direction_penalty2
        
        reward3 = 1.-.003*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        #print(distance_penalty)
        #print(direction_penalty)
        #return reward2
        
        reward = np.tanh(reward)
        if(distance < 2):
            reward += 5
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state