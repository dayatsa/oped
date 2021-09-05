#!/usr/bin/env python2

from __future__ import print_function
import rospy
import roslib; roslib.load_manifest('oped_teleopp')
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os
import numpy as np
from datetime import datetime
from quadruped_controller import *
from floor_controller import *
from collections      import deque


class Agent():
    def __init__(self, state_size, action_size, episodes):
        self.is_weight_backup   = True
        self.WEIGHT_BACKUP      = "/home/dayatsa/model_editor_models/oped/src/oped/oped_teleopp/model/model_"
        self.WEIGHT_LOAD_Y      = "/home/dayatsa/model_editor_models/oped/src/oped/oped_teleopp/model/model_y_05-09-2021_20:15.npy"
        self.WEIGHT_LOAD_X      = "/home/dayatsa/model_editor_models/oped/src/oped/oped_teleopp/model/model_x_05-09-2021_11:51.npy"
        self.STATE_SIZE         = state_size
        self.ACTION_SIZE        = action_size
        self.LEARNING_RATE      = 0.1
        self.GAMMA              = 0.95
        self.EXPLORATION_MIN    = 0.01
        self.START_EXPLORATION_DECAY = 1
        self.END_EXPLORATION_DECAY = episodes//2
        self.EXPLORATION_DECAY  = 1.0/float(self.END_EXPLORATION_DECAY - self.START_EXPLORATION_DECAY)
        print("Exploration decay: {} , {} , {}".format(self.START_EXPLORATION_DECAY, self.END_EXPLORATION_DECAY, self.EXPLORATION_DECAY))
        self.exploration_rate   = 1.0
        self.DISCRETE_OS_SIZE   = [80, 60]
        self.DISCRETE_OS_SIZE_Q   = [81, 61]
        self.observation_space_high = np.array([40.0, 15.0])
        self.observation_space_low = np.array([-40.0, -15.0])
        self.discrete_os_win_size = (self.observation_space_high - self.observation_space_low)/self.DISCRETE_OS_SIZE
        print("Discrete: ", self.discrete_os_win_size)
        self.q_table_y            = self.buildModel(self.WEIGHT_LOAD_Y)
        self.q_table_x            = self.buildModel(self.WEIGHT_LOAD_X)


    def buildModel(self, directory):
        if not self.is_weight_backup:
            q_table = np.random.uniform(low=0, high=1, size=(self.DISCRETE_OS_SIZE_Q + [self.ACTION_SIZE]))
        else:
            print("\n\n================LOADING Q-TABLE===============\n\n")
            print(directory)
            q_table = np.load(directory)
            self.exploration_rate = self.EXPLORATION_MIN
        print(q_table.shape)
        return q_table
    

    def getDiscreteState(self, state):
        if (state[0] > 40):
            state[0] = 40
        elif (state[0] < -40):
            state[0] = -40

        if (state[1] > 15.0):
            state[1] = 15.0
        elif (state[1] < -15.0):
            state[1] = -15.0

        discrete_state = (state - self.observation_space_low)/self.discrete_os_win_size
        return tuple(discrete_state.astype(np.int))


    def saveModel(self):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H:%M")
        np.save(self.WEIGHT_BACKUP + "y_" + dt_string + ".npy", self.q_table_y)
        # np.save(self.WEIGHT_BACKUP + "x_" + dt_string + ".npy", self.q_table_x)


    def action(self, state, is_y):
        if np.random.rand() > self.exploration_rate:
            if is_y:
                return np.argmax(self.q_table_y[state])
            else:
                return np.argmax(self.q_table_x[state])
        return np.random.randint(0, self.ACTION_SIZE)


    def updateModel(self, discrete_state, new_discrete_state, action, reward, is_y):
        if is_y:
            max_future_q = np.max(self.q_table_y[new_discrete_state])
            current_q = self.q_table_y[discrete_state + (action,)]
            new_q = (1 - self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (reward + self.GAMMA * max_future_q)
            self.q_table_y[discrete_state + (action,)] = new_q
        else:
            max_future_q = np.max(self.q_table_x[new_discrete_state])
            current_q = self.q_table_x[discrete_state + (action,)]
            new_q = (1 - self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (reward + self.GAMMA * max_future_q)
            self.q_table_x[discrete_state + (action,)] = new_q


    def updateExplorationRate(self, episode):
        if self.END_EXPLORATION_DECAY >= episode >= self.START_EXPLORATION_DECAY:
            if self.exploration_rate > self.EXPLORATION_MIN:
                self.exploration_rate -= self.EXPLORATION_DECAY