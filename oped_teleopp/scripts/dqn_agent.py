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
import tensorflow as tf
from datetime import datetime
from quadruped_controller import *
from floor_controller import *
from collections      import deque
from tensorflow       import keras


class DQNAgent():
    def __init__(self, state_size, action_size, episodes):
        self.is_weight_backup   = True
        self.WEIGHT_BACKUP      = "/home/dayatsa/model_editor_models/oped/src/oped/oped_teleopp/model/model_"
        self.WEIGHT_LOAD        = "/home/dayatsa/model_editor_models/oped/src/oped/oped_teleopp/model/model_01-09-2021_11:53.h5"
        self.STATE_SIZE         = state_size
        self.ACTION_SIZE        = action_size
        self.LEARNING_RATE      = 0.001
        self.GAMMA              = 0.95
        self.EXPLORATION_MIN    = 0.01
        self.START_EXPLORATION_DECAY = 1
        self.END_EXPLORATION_DECAY = episodes*3//4#episodes//2
        self.EXPLORATION_DECAY  = 1.0/float(self.END_EXPLORATION_DECAY - self.START_EXPLORATION_DECAY)
        print("Exploration decay: {} , {} , {}".format(self.START_EXPLORATION_DECAY, self.END_EXPLORATION_DECAY, self.EXPLORATION_DECAY))
        self.exploration_rate   = 1.0
        self.memory             = deque(maxlen=300)
        self.model              = self.buildModel()


    def buildModel(self):
        # Neural Net for Deep-Q learning Model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.STATE_SIZE, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(self.ACTION_SIZE, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.LEARNING_RATE))
        
        if self.is_weight_backup is True:
            print("\n\n=============Loading saved model==============\n\n")
            model.load_weights(self.WEIGHT_LOAD)
            # self.exploration_rate = self.EXPLORATION_MIN
        return model


    def saveModel(self):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H:%M")
        self.model.save(self.WEIGHT_BACKUP + dt_string + ".h5")


    def action(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.ACTION_SIZE)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def replay(self, sample_batch_size, epsilon):
        if len(self.memory) < sample_batch_size:
            print("SAMPLE BATCH SIZE")
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + self.GAMMA * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        print("Exploration rate: {}".format(self.exploration_rate))
        if self.END_EXPLORATION_DECAY >= epsilon >= self.START_EXPLORATION_DECAY:
            if self.exploration_rate > self.EXPLORATION_MIN:
                self.exploration_rate -= self.EXPLORATION_DECAY
                # print("Exploration decay: {} , {}".format(self.exploration_rate, self.EXPLORATION_DECAY))