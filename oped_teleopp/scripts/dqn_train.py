#!/usr/bin/env python2

import rospy
import roslib; roslib.load_manifest('oped_teleopp')
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os
import numpy as np
from dqn_agent import *
from quadruped_controller import *
from floor_controller import *


class OpedTrainer:
    def __init__(self):
        self.SAMPLE_BATCH_SIZE = 32
        self.EPISODES          = 10000

        self.oped              = Quadruped()
        self.floor             = Floor()
        self.STATE_SPACE       = self.oped.STATE_SPACE
        self.ACTION_SIZE       = self.oped.ACTION_N
        self.MAX_EPISODE       = self.oped.MAX_EPISODE
        self.agent             = DQNAgent(self.STATE_SPACE, self.ACTION_SIZE)        
        self.floor_position_x  = 0
        self.floor_position_y  = 0
        self.set_point_floor_x_adder = 0
        self.set_point_floor_y_adder = 0

    
    def getFloorSetPoint(self):
        self.floor_position_x = 0
        self.floor_position_y = 0
        self.set_point_floor_x_adder = np.random.uniform(floor.MIN_DEGREE, floor.MAX_DEGREE)/self.MAX_EPISODE
        self.set_point_floor_y_adder = np.random.uniform(floor.MIN_DEGREE, floor.MAX_DEGREE)/self.MAX_EPISODE
    

    def resetEnvironment(self):
        self.floor.setInitialPosition()
        self.oped.setInitialPosition()
        rospy.sleep(0.5)
        self.floor.resetWorld()
        rospy.sleep(0.5)
        return self.oped.getState()

    
    def floorStep(self):
        self.floor.setPosition(floor_position_x, floor_position_y)
        self.floor_position_x += self.set_point_floor_x_adder
        self.floor_position_y += self.set_point_floor_y_adder


    def run(self):
        try:
            for index_episode in range(self.EPISODES):
                state = self.resetEnvironment()
                state = np.reshape(state, [1, self.state_size])

                done = False
                episode_reward = 0
                while not done:
                    action = self.agent.action(state)

                    next_state, reward, done = self.oped.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.floorStep()
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward
                print("Episode {}# Score: {}".format(index_episode, episode_reward))
                self.agent.replay(self.SAMPLE_BATCH_SIZE)
        finally:
            self.agent.saveModel()


if __name__ == "__main__":
    rospy.init_node('train', anonymous=True)
    rate = rospy.Rate(25) # 
    oped_agent = OpedTrainer()
    oped_agent.run()