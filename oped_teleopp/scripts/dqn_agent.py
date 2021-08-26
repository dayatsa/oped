#!/usr/bin/env python2

import rospy
import roslib; roslib.load_manifest('oped_teleopp')
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os
import numpy as np
from quadruped_controller import *
from floor_controller import *
from collections      import deque
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam


class DQNAgent():
    def __init__(self, state_size, action_size):
        self.WEIGHT_BACKUP      = "oped_weight.h5"
        self.STATE_SIZE         = state_size
        self.ACTION_SIZE        = action_size
        self.LEARNING_RATE      = 0.001
        self.GAMMA              = 0.95
        self.EXPLORATION_MIN    = 0.01
        self.EXPLORATION_DECAY  = 0.995
        self.exploration_rate   = 1.0
        self.memory             = deque(maxlen=2000)
        self.model              = self.buildModel()


    def buildModel(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.STATE_SIZE, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.ACTION_SIZE, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.LEARNING_RATE))
        
        if os.path.isfile(self.WEIGHT_BACKUP):
            model.load_weights(self.WEIGHT_BACKUP)
            self.exploration_rate = self.EXPLORATION_MIN
        return model


    def saveModel(self):
        self.model.save(self.WEIGHT_BACKUP)


    def action(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.ACTION_SIZE)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay