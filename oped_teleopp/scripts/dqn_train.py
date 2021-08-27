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
import json
from datetime import datetime
from dqn_agent import *
from quadruped_controller import *
from floor_controller import *


class OpedTrainer:
    def __init__(self):
        self.SAMPLE_BATCH_SIZE = 32
        self.EPISODES          = 1000

        self.oped              = Quadruped()
        self.floor             = Floor()
        self.STATE_SPACE       = self.oped.STATE_SPACE
        self.ACTION_SIZE       = self.oped.ACTION_N
        self.MAX_EPISODE       = self.oped.MAX_EPISODE
        self.agent             = DQNAgent(self.STATE_SPACE, self.ACTION_SIZE, self.EPISODES)        
        self.floor_position_x  = 0
        self.floor_position_y  = 0
        self.set_point_floor_x_adder = 0
        self.set_point_floor_y_adder = 0

    
    def getFloorSetPoint(self):
        self.floor_position_x = 0
        self.floor_position_y = 0
        self.set_point_floor_x_adder = np.random.uniform(self.floor.MIN_DEGREE, self.floor.MAX_DEGREE)/self.MAX_EPISODE
        self.set_point_floor_y_adder = np.random.uniform(self.floor.MIN_DEGREE, self.floor.MAX_DEGREE)/self.MAX_EPISODE
    

    def resetEnvironment(self):
        self.floor.setInitialPosition()
        self.oped.setInitialPosition()
        rospy.sleep(0.5)
        self.oped.resetWorld()
        rospy.sleep(0.5)
        self.getFloorSetPoint()
        return self.oped.getState()

    
    def floorStep(self):
        self.floor.setPosition(self.floor_position_x, self.floor_position_y)
        self.floor_position_x += self.set_point_floor_x_adder
        self.floor_position_y += self.set_point_floor_y_adder

    
    def saveRewardValue(self, my_dict):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H:%M")

        path = "/home/dayatsa/model_editor_models/oped/src/oped/oped_teleopp/rewards/reward" + dt_string + ".json"
        with open(path, 'w') as fp:
            json.dump(my_dict, fp)



    def run(self):
        aggr_ep_rewards = {'ep': [], 'rewards': []}
        try:
            for index_episode in range(self.EPISODES):
                print("Reset Environment")
                state = self.resetEnvironment()
                state = np.reshape(state, [1, self.STATE_SPACE])

                done = False
                episode_reward = 0

                while not done:
                    action = self.agent.action(state)
                    next_state, reward, done = self.oped.step(action)
                    next_state = np.reshape(next_state, [1, self.STATE_SPACE])
                    self.floorStep()
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward
                    rate.sleep()

                print("Episode {} # Reward: {}".format(index_episode, episode_reward))
                aggr_ep_rewards['ep'].append(index_episode)
                aggr_ep_rewards['rewards'].append(episode_reward)
                self.agent.replay(self.SAMPLE_BATCH_SIZE)

        finally:
            self.agent.saveModel()
            self.saveRewardValue(aggr_ep_rewards)
            plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['rewards'], label="rewards")
            plt.show()


if __name__ == "__main__":
    rospy.init_node('train', anonymous=True)
    rate = rospy.Rate(25) # 
    oped_agent = OpedTrainer()
    oped_agent.run()