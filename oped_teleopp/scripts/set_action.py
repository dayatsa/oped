#!/usr/bin/env python2


from quadruped_controller import *
from floor_controller import *
import rospy
import roslib; roslib.load_manifest('oped_teleop')


# leg = Quadruped()
leg = Floor()
MIN_DEGREE = -15
MAX_DEGREE = 15 #57.2958


def talker():    
    rospy.init_node('joint_states', anonymous=True)

    rate = rospy.Rate(1) # 10hz
    lift = True
    x = 0

    while not rospy.is_shutdown():    
        if (lift == True):
            x += 1
        else:
            x -= 1
        if(x>MAX_DEGREE):
            lift = False
        elif(x<MIN_DEGREE) :
            lift = True    

        leg.setPosition(x, 0)
        rospy.loginfo(x)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass