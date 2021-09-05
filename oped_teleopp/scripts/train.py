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
from agent import *
from quadruped_controller import *
from floor_controller import *


class OpedTrainer:
    def __init__(self):
        self.SAMPLE_BATCH_SIZE = 20
        self.EPISODES          = 10000

        self.oped              = Quadruped()
        self.floor             = Floor()
        self.STATE_SPACE       = self.oped.STATE_SPACE
        self.ACTION_SIZE       = self.oped.ACTION_N
        self.MAX_EPISODE       = self.oped.MAX_EPISODE
        self.STATS_EVERY       = 20
        self.agent             = Agent(self.STATE_SPACE, self.ACTION_SIZE, self.EPISODES)        
        self.floor_position_x  = 0
        self.floor_position_y  = 0
        self.set_point_floor_x_adder = 0
        self.set_point_floor_y_adder = 0
        self.now               = datetime.now()
        self.dt_start_string   = self.now.strftime("%d-%m-%Y_%H:%M")

    
    def getFloorSetPoint(self):
        self.floor_position_x = 0
        self.floor_position_y = 0

        # if np.random.rand() < 0.5:
        #     self.set_point_floor_x_adder = np.random.uniform(5, self.floor.MAX_DEGREE)/self.MAX_EPISODE
        # else:
        #     self.set_point_floor_x_adder = np.random.uniform(self.floor.MIN_DEGREE, -5)/self.MAX_EPISODE
        
        if np.random.rand() < 0.5:
            self.set_point_floor_y_adder = np.random.uniform(5, self.floor.MAX_DEGREE)/self.MAX_EPISODE
        else:
            self.set_point_floor_y_adder = np.random.uniform(self.floor.MIN_DEGREE, -5)/self.MAX_EPISODE


    def resetEnvironment(self):
        self.floor.setInitialPosition()
        self.oped.setInitialPosition()
        rospy.sleep(0.2)
        self.oped.resetWorld()
        rospy.sleep(0.5)
        self.getFloorSetPoint()
        return self.oped.getState()

    
    def floorStep(self):
        self.floor.setPosition(self.floor_position_y, self.floor_position_x)
        self.floor_position_x += self.set_point_floor_x_adder
        self.floor_position_y += self.set_point_floor_y_adder

    
    def saveRewardValue(self, my_dict):
        self.now = datetime.now()
        dt_string = self.now.strftime("%d-%m-%Y_%H:%M")

        dict_model   = {"lr":self.agent.LEARNING_RATE,
                        "gamma":self.agent.GAMMA,
                        "move_step":self.oped.MOVE_STEP,
                        "limit_upright":self.oped.LIMIT_UPRIGHT,
                        "action_size":self.oped.ACTION_N,
                        "start_date":self.dt_start_string,
                        "end_date":dt_string,
                        "rewards":my_dict}

        path = "/home/dayatsa/model_editor_models/oped/src/oped/oped_teleopp/rewards/reward" + dt_string + ".json"
        with open(path, 'w') as fp:
            json.dump(dict_model, fp)



    def run(self):
        ep_rewards = []
        aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
        try:
            for index_episode in range(self.EPISODES):
                print()
                # print("Reset Environment")
                state = self.resetEnvironment()
                discrete_state = self.agent.getDiscreteState(state)
                print("state: ", state)
                print("disecrete_state: ", discrete_state)

                done = False
                episode_reward = 0
                index = 0 
                while not done:
                    action = self.agent.action(discrete_state)

                    next_state, reward, done = self.oped.step(action)
                    new_discrete_state = self.agent.getDiscreteState(next_state)
                    episode_reward += reward

                    self.floorStep()
                    # print(next_state)
                    index += 1
                    if not done:
                        self.agent.updateModel(discrete_state, new_discrete_state, action, reward)
                    
                    rate.sleep()    
                    discrete_state = new_discrete_state
                
                self.agent.updateExplorationRate(index_episode)
                print("Episode {}, index: {}, # Reward: {}".format(index_episode, index, episode_reward))
                print("Exploration: {}, x: {}, y: {}".format(self.agent.exploration_rate, self.floor_position_x, self.floor_position_y))
               
                ep_rewards.append(episode_reward)
                if not index_episode % self.STATS_EVERY:
                    average_reward = sum(ep_rewards[-self.STATS_EVERY:])/self.STATS_EVERY
                    aggr_ep_rewards['ep'].append(index_episode)
                    aggr_ep_rewards['avg'].append(average_reward)
                    aggr_ep_rewards['max'].append(max(ep_rewards[-self.STATS_EVERY:]))
                    aggr_ep_rewards['min'].append(min(ep_rewards[-self.STATS_EVERY:]))
                    print("Episode: {}, average reward: {}".format(index_episode, average_reward))
                    ep_rewards = []

        finally:
            self.agent.saveModel()
            # reward_cumulative = {"aggr_rewards":aggr_ep_rewards, "rewards":ep_rewards}
            self.saveRewardValue(aggr_ep_rewards)

            plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
            plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
            plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
            plt.legend(loc=4)
            plt.show()


if __name__ == "__main__":
    rospy.init_node('train', anonymous=True)
    rate = rospy.Rate(25) # 
    oped_agent = OpedTrainer()
    oped_agent.run()