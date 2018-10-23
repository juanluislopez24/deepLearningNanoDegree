import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state, episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        epsilon = 1 / (episode)
        if epsilon > 1:
            epsilon =1
        elif epsilon < 0.00025:
            epsilon = 0.00025# epsilon *2
        max_sa = None
        policy = np.ones(self.nA)/self.nA * epsilon
        max_actions = np.array((np.argwhere(self.Q[state] == np.amax(self.Q[state] )).flatten().tolist()))
        if(len(max_actions)>1):
            max_sa = np.random.choice(max_actions)
        else:
            max_sa = np.argmax(self.Q[state])
        policy[max_sa] = 1 - epsilon + (epsilon/self.nA)
        action = np.random.choice(np.arange(self.nA), p=policy)
        return action

    def step(self, state, action, reward, next_state, done, alpha = 0.05, gamma = 0.95):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        #self.Q[state][action] = self.Q[state][action] + (alpha*(reward + gamma*(np.dot(self.Q[next_state], policy)) - Q[state][action]))
        if(self.Q[next_state][action] == 0):
            self.Q[state][action] = self.Q[state][action] + (alpha*(reward - self.Q[state][action]))
        else:
            self.Q[state][action] = self.Q[state][action] + (alpha*(reward + gamma*(np.max(self.Q[next_state])) - self.Q[state][action]))   
            
    