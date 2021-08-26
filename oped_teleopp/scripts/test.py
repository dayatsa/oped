#!/usr/bin/env python2

import rospy
import roslib; roslib.load_manifest('oped_teleopp')
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from quadruped_controller import *
from floor_controller import *


oped = Quadruped()

def test():
    rospy.init_node('test', anonymous=True)
    rate = rospy.Rate(120) # 10hz
    rospy.loginfo("Loading q-table")
    # oped.setInitialPosition()
    oped.resetWorld()
    for i in range(-10,60):
        oped.setPosition(i,i,i,i)
        rate.sleep()
    rospy.loginfo("done")


if __name__ == '__main__':
    try:
        test()
    except rospy.ROSInterruptException:
        pass