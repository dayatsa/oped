#!/usr/bin/env python2


from __future__ import print_function
from std_msgs.msg import String
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import SpawnModel, DeleteModel
from sensor_msgs.msg import Imu
import roslib; roslib.load_manifest('oped_teleopp')
import numpy as np
import rospy


class Joint(object):
    def __init__(self):
        self.RAD_PER_DEG = 0.017453293
        self.DEG_PER_RAD = 57.29577951
        self.MIN_DEGREE = -15
        self.MAX_DEGREE = 15    #57.2958
        self.MOVE_STEP = 8

        self.joint1 = 0
        self.joint2 = 0
        self.joint1_states_publisher = rospy.Publisher('/floor/joint1_position_controller/command', Float64, queue_size=1)
        self.joint2_states_publisher = rospy.Publisher('/floor/joint2_position_controller/command', Float64, queue_size=1)


    def degreeToRad(self, joint1, joint2):
        joint1 = joint1 * self.RAD_PER_DEG
        joint2 = joint2 * self.RAD_PER_DEG
        return joint1, joint2


    def radToDegree(self, joint1, joint2):
        joint1 = joint1 * self.DEG_PER_RAD
        joint2 = joint2 * self.DEG_PER_RAD
        return joint1, joint2

    
    def addPosition(self, joint1, joint2):
        self.joint1 = self.joint1 + joint1
        self.joint2 = self.joint2 + joint2

        if self.joint1 > self.MAX_DEGREE:
            self.joint1 = self.joint1 - joint1
        if self.joint2 > self.MAX_DEGREE:
            self.joint2 = self.joint2 - joint2

        if self.joint1 < self.MIN_DEGREE:
            self.joint1 = self.joint1 + joint1
        if self.joint2 < self.MIN_DEGREE:
            self.joint2 = self.joint2 + joint2        

        self.publishPosition(self.joint1, self.joint2)


    def setPosition(self, joint1, joint2):
        self.joint1 = joint1
        self.joint2 = joint2

        if self.joint1 > self.MAX_DEGREE:
            self.joint1 = self.MAX_DEGREE
        if self.joint2 > self.MAX_DEGREE:
            self.joint2 = self.MAX_DEGREE

        if self.joint1 < self.MIN_DEGREE:
            self.joint1 = self.MIN_DEGREE
        if self.joint2 < self.MIN_DEGREE:
            self.joint2 = self.MIN_DEGREE

        self.publishPosition(self.joint1, self.joint2)


    def setInitialPosition(self):
        self.joint1 = 0
        self.joint2 = 0
        self.setPosition(0, 0)


    def publishPosition(self, joint1_, joint2_):
        joint1, joint2 = self.degreeToRad(joint1_, joint2_)   

        joints_msg1 = Float64()
        joints_msg2 = Float64()

        joints_msg1.data = joint1
        joints_msg2.data = joint2

        self.joint1_states_publisher.publish(joints_msg1)
        self.joint2_states_publisher.publish(joints_msg2)



class Floor(Joint) : 
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.MODEL_URDF = '/home/dayatsa/model_editor_models/oped/src/floor/floor_description/urdf/floor.urdf'
        Joint.__init__(self)


    def __str__(self):
        return str(self.x + ", " + self.y + ", " + self.z)


    def resetWorld(self):
        rospy.wait_for_service('/gazebo/reset_world')
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_world()

    
    def spawnURDF(self, name, description_xml, ns, pose, reference_frame='world'):
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        try:
            spawn_urdf_clint = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
            resp_urdf = spawn_urdf_clint(name, description_xml, ns, pose, reference_frame)
        except rospy.ServiceException as e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e)) 


    def deleteURDF(self, model):
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            resp_delete = delete_model(model)

        except rospy.ServiceException as e:
            print("Delete Model service call failed: {0}".format(e)) 
