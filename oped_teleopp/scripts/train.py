#!/usr/bin/env python2

import rospy
import roslib; roslib.load_manifest('oped_teleop')
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from quadruped_controller import *
from floor_controller import *


HM_EPISODES = 25000
MOVE_PENALTY = 1  
FALL_PENALTY = -300  
STABILITY_REWARD = 25  
SHOW_EVERY = 10
EPS_DECAY = 0.9999 

start_q_table = None  

LEARNING_RATE = 0.1
DISCOUNT = 0.95

MIN_DEGREE = -11
MAX_DEGREE = 68 #57.2958
IMU_MIN_DEGREE = -15*3/2
IMU_MAX_DEGREE = 15*3/2 #57.2958
LIMIT_IMU = 15

start_q_table = "/home/dayatsa/model_editor_models/oped/src/oped/oped_teleop/src/qtable.pickle"

oped = Quadruped()
floor = Floor()

def getData():
    data = oped.getInfo()
    lf = data['lf']
    lh = data['lh']
    rf = data['rf']
    rh = data['rh']
    x = data['x']
    y = data['y']
    z = data['z']
    return lf, lh, rf, rh, x, y, z


def train():
    rospy.init_node('train', anonymous=True)
    rate = rospy.Rate(120) # 10hz
    rospy.loginfo("Loading q-table")

    if start_q_table is None:
        # initialize the q-table#
        q_table = {}
        for leg1 in range(MIN_DEGREE+3, MAX_DEGREE, 8):
            for leg2 in range(MIN_DEGREE+3, MAX_DEGREE, 8):
                for leg3 in range(MIN_DEGREE+3, MAX_DEGREE, 8):
                    for leg4 in range(MIN_DEGREE+3, MAX_DEGREE, 8):
                        for pitch in range(-LIMIT_IMU, LIMIT_IMU+1, 3):
                            for roll in range(-LIMIT_IMU, LIMIT_IMU+1, 3):
                                q_table[leg1,leg2,leg3,leg4,pitch,roll] = [np.random.uniform(-5, 0) for i in range(8)]

    # else:
    #     with open(start_q_table, "rb") as f:
    #         q_table = pickle.load(f)

    epsilon = 0.5  
    episode_rewards = []
    rospy.loginfo("Start training...")

    for episode in range(HM_EPISODES):

        rospy.loginfo("Resetting World")
        floor.setInitialPosition()
        oped.setInitialPosition()
        floor.resetWorld()
        oped.resetWorld()
        rospy.sleep(1)


        floor_position = 0
        set_point_floor = np.random.uniform(floor.MIN_DEGREE, floor.MAX_DEGREE)
        set_point_floor_adder = set_point_floor/300

        # if episode % SHOW_EVERY == 0:
        rospy.loginfo("on #" + str(episode) + ", epsilon is " + str(epsilon))
        # rospy.loginfo(str(SHOW_EVERY) + " ep mean: " + str(np.mean(episode_rewards[-SHOW_EVERY:])))

        episode_reward = 0
        for i in range(300):
            x = np.random.randint(0,9)

            floor.setPosition(floor_position,0)
            floor_position += set_point_floor_adder

            action = oped.action(x)
            lf, lh, rf, rh, x, y, z = getData()

            if ((x < IMU_MIN_DEGREE or x > IMU_MAX_DEGREE) or (y < IMU_MIN_DEGREE or y > IMU_MAX_DEGREE)):
                rospy.loginfo("Break")
                break
            
            # obs = (player-food, player-enemy)
            # #print(obs)
            # if np.random.random() > epsilon:
            #     # GET THE ACTION
            #     action = np.argmax(q_table[obs])
            # else:
            #     action = np.random.randint(0, 4)
            # # Take the action!
            # player.action(action)

            # #### MAYBE ###
            # #enemy.move()
            # #food.move()
            # ##############

            # if player.x == enemy.x and player.y == enemy.y:
            #     reward = -ENEMY_PENALTY
            # elif player.x == food.x and player.y == food.y:
            #     reward = FOOD_REWARD
            # else:
            #     reward = -MOVE_PENALTY
            # ## NOW WE KNOW THE REWARD, LET'S CALC YO
            # # first we need to obs immediately after the move.
            # new_obs = (player-food, player-enemy)
            # max_future_q = np.max(q_table[new_obs])
            # current_q = q_table[obs][action]

            # if reward == FOOD_REWARD:
            #     new_q = FOOD_REWARD
            # else:
            #     new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # q_table[obs][action] = new_q

            # if show:
            #     env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            #     env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
            #     env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue
            #     env[enemy.x][enemy.y] = d[ENEMY_N]  # sets the enemy location to red
            #     img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            #     img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            #     cv2.imshow("image", np.array(img))  # show it!
            #     if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
            #         if cv2.waitKey(500) & 0xFF == ord('q'):
            #             break
            #     else:
            #         if cv2.waitKey(1) & 0xFF == ord('q'):
            #             break

            # episode_reward += reward
            # if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            #     break

            rate.sleep()

        #print(episode_reward)
        episode_rewards.append(episode_reward)
        epsilon *= EPS_DECAY

    # moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

    # plt.plot([i for i in range(len(moving_avg))], moving_avg)
    # plt.ylabel(f"Reward {SHOW_EVERY}ma")
    # plt.xlabel("episode #")
    # plt.show()

    # with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    #     pickle.dump(q_table, f)

    # with open("/home/dayatsa/model_editor_models/oped/src/oped/oped_teleop/src/qtable.pickle", "wb") as f:
    #     pickle.dump(q_table, f)


if __name__ == '__main__':
    try:
        train()
    except rospy.ROSInterruptException:
        pass