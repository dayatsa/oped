#!/usr/bin/env python2

from __future__ import print_function
import numpy as np

class Agent():
    def __init__(self):
        self.ACTION_SIZE = 2
        self.exploration_rate   = 1.0
        self.DISCRETE_OS_SIZE   = [80, 60]
        self.DISCRETE_OS_SIZE_Q   = [81, 61]
        self.observation_space_high = np.array([40.0, 15.0])
        self.observation_space_low = np.array([-40.0, -15.0])
        self.discrete_os_win_size = (self.observation_space_high - self.observation_space_low)/self.DISCRETE_OS_SIZE
        print("Discrete: ", self.discrete_os_win_size)

        self.q_table            = self.buildModel()


    def buildModel(self):
        q_table = np.random.uniform(low=0, high=1, size=(self.DISCRETE_OS_SIZE_Q + [self.ACTION_SIZE]))
        print(q_table.shape)
        return q_table


    def getDiscreteState(self, state):
        if (state[1] > 15.0):
            state[1] = 15.0
        elif (state[1] < -15.0):
            state[1] = -15.0
        discrete_state = (state - self.observation_space_low)/self.discrete_os_win_size
        return tuple(discrete_state.astype(np.int))


agen = Agent()
state = agen.getDiscreteState([40,15])
print(state)
print(agen.q_table[state])
