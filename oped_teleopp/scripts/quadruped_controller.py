#!/usr/bin/env python2


from __future__ import print_function
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import SpawnModel, DeleteModel
from sensor_msgs.msg import Imu
import roslib; roslib.load_manifest('oped_teleopp')
import numpy as np
import rospy


class MyImu(object):
    def __init__(self):
        self.orientation_x = 0
        self.orientation_y = 0
        self.orientation_z = 0
        self.DEG_PER_RAD = 57.29577951
        self.LIMIT_UPRIGHT = 0.5
        self.IMU_MIN_DEGREE = -10
        self.IMU_MAX_DEGREE = 10
        imu_subsriber = rospy.Subscriber("/imu_oped/data", Imu, self.imuCallback)


    def imuCallback(self, data):
        self.orientation_x = data.orientation.x * 2 * self.DEG_PER_RAD
        self.orientation_y = data.orientation.y * 2 * self.DEG_PER_RAD
        self.orientation_z = data.orientation.z * 2 * self.DEG_PER_RAD

    
    def getImuData(self):
        return self.orientation_x, self.orientation_y, self.orientation_z



class Leg(object):
    def __init__(self):
        self.RAD_PER_DEG = 0.017453293
        self.MIN_DEGREE = -11.4592
        # self.MAX_DEGREE = 68.7549 #57.2958
        self.MAX_DEGREE = 97.4 #57.2958
        self.MOVE_STEP = 0.29
        self.MIDDLE_POSITION = 42.975

        self.lf = 0.0
        self.lh = 0.0
        self.rf = 0.0
        self.rh = 0.0
        self.leg_y = 0.0
        self.leg_x = 0.0
        self.last_lf = 0.0
        self.last_lh = 0.0
        self.last_rf = 0.0
        self.last_rh = 0.0
        self.leg_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_names = ['lf_hip_joint', 'lf_upper_leg_joint', 'lf_lower_leg_joint', 'lh_hip_joint', 'lh_upper_leg_joint', 'lh_lower_leg_joint', 'rf_hip_joint', 'rf_upper_leg_joint', 'rf_lower_leg_joint', 'rh_hip_joint', 'rh_upper_leg_joint', 'rh_lower_leg_joint']
        
        self.joint_group_publisher = rospy.Publisher('/oped/joint_group_position_controller/command', JointTrajectory, queue_size=1)
        # rospy.init_node('joint_states', anonymous=True)


    def degreeToRad(self, lf, lh, rf, rh):
        lf = lf * self.RAD_PER_DEG
        lh = lh * self.RAD_PER_DEG
        rf = rf * self.RAD_PER_DEG
        rh = rh * self.RAD_PER_DEG
        return lf, lh, rf, rh


    def addPosition(self, step_y, step_x):

        self.leg_y = self.leg_y + step_y*self.MOVE_STEP
        self.leg_x = self.leg_x + step_x*self.MOVE_STEP

        self.lf = self.MIDDLE_POSITION + self.leg_y - self.leg_x
        self.lh = self.MIDDLE_POSITION - self.leg_y - self.leg_x
        self.rf = self.MIDDLE_POSITION + self.leg_y + self.leg_x
        self.rh = self.MIDDLE_POSITION - self.leg_y + self.leg_x
        # print(self.lf, self.lh, self.rf, self.rh)

        if (self.lf > self.MAX_DEGREE or self.lf < self.MIN_DEGREE or self.lh > self.MAX_DEGREE or self.lh < self.MIN_DEGREE or self.rf > self.MAX_DEGREE or self.rf < self.MIN_DEGREE or self.rh > self.MAX_DEGREE or self.rh < self.MIN_DEGREE):
            self.lf = self.last_lf
            self.lh = self.last_lh
            self.rf = self.last_rf
            self.rh = self.last_rh

        if self.lf > self.MAX_DEGREE:
            self.lf = self.MAX_DEGREE
        if self.lh > self.MAX_DEGREE:
            self.lh = self.MAX_DEGREE
        if self.rf > self.MAX_DEGREE:
            self.rf = self.MAX_DEGREE
        if self.rh > self.MAX_DEGREE:
            self.rh = self.MAX_DEGREE

        if self.lf < self.MIN_DEGREE:
            self.lf = self.MIN_DEGREE
        if self.lh < self.MIN_DEGREE:
            self.lh = self.MIN_DEGREE
        if self.rf < self.MIN_DEGREE:
            self.rf = self.MIN_DEGREE
        if self.rh < self.MIN_DEGREE:
            self.rh = self.MIN_DEGREE          

        self.last_lf = self.lf
        self.last_lh = self.lh
        self.last_rf = self.rf
        self.last_rh = self.rh

        # print(self.lf, self.lh, self.rf, self.rh)
        
        self.setPosition(self.lf, self.lh, self.rf, self.rh)


    def setPosition(self, lf_, lh_, rf_, rh_):
        lf, lh, rf, rh = self.degreeToRad(lf_, lh_, rf_, rh_)      

        lf_hip = 0
        lf_upper = lf
        lf_lower = -lf*3/2

        lh_hip = 0
        lh_upper = -lh
        lh_lower = lh*3/2

        rf_hip = 0
        rf_upper = rf
        rf_lower = -rf*3/2

        rh_hip = 0
        rh_upper = -rh
        rh_lower = rh*3/2

        self.leg_position = [lf_hip, lf_upper, lf_lower, lh_hip, lh_upper, lh_lower, rf_hip, rf_upper, rf_lower, rh_hip, rh_upper, rh_lower]
        self.publishPosition()

    
    def setInitialPosition(self):
        self.leg_y = 0.0
        self.leg_x = 0.0
        self.lf = self.MIDDLE_POSITION
        self.lh = self.MIDDLE_POSITION
        self.rf = self.MIDDLE_POSITION
        self.rh = self.MIDDLE_POSITION
        self.setPosition(self.lf, self.lh, self.rf, self.rh)


    def publishPosition(self):
        joints_msg = JointTrajectory()
        joints_msg.header.stamp = rospy.Time.now()
        joints_msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Duration(1.0 / 60.0)
        point.positions = self.leg_position #position

        joints_msg.points = [point]
        # rospy.loginfo(joints_msg)
        # rospy.loginfo("---")

        self.joint_group_publisher.publish(joints_msg)


    def getLegPosition(self):
        return self.lf, self.lh, self.rf, self.rh



class Quadruped(Leg, MyImu) : 
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.8
        self.MODEL_URDF = '/home/dayatsa/model_editor_models/oped/src/oped/oped_description/urdf/oped.urdf'
        self.ACTION_N = 3
        self.STATE_SPACE = 2
        self.MAX_EPISODE = 500
        self.episode_step = 0
        Leg.__init__(self)
        MyImu.__init__(self)


    def __str__(self):
        return str(self.x + ", " + self.y + ", " + self.z)


    def step(self, choice1, choice2):
        '''
        Gives us 3 total movement options.
        '''
        self.episode_step += 1

        step_y = 0
        step_x = 0

        if choice1 == 0:
            step_y = 0
        elif choice1 == 1:
            step_y = -1
        elif choice1 == 2:
            step_y = 1

        if choice2 == 0:
            step_x = 0
        elif choice2 == 1:
            step_x = -1
        elif choice2 == 2:
            step_x = 1

        self.addPosition(step_y, step_x)  

        new_state_imu = self.getImuData()
        y = new_state_imu[1]
        x = new_state_imu[0]

        #reward
        reward_y = 0
        reward_x = 0

        if y > -self.LIMIT_UPRIGHT and y < self.LIMIT_UPRIGHT:
            reward_y += 100
        else:
            if y < 0:
                reward_y += y
            else:
                reward_y -= y
        
        if x > -self.LIMIT_UPRIGHT and x < self.LIMIT_UPRIGHT:
            reward_x += 100
        else:
            if x < 0:
                reward_x += x
            else:
                reward_x -= x

        done = False
        if (x < self.IMU_MIN_DEGREE or x > self.IMU_MAX_DEGREE):
            done = True
            rospy.loginfo("x imu")
        if (y < self.IMU_MIN_DEGREE or y > self.IMU_MAX_DEGREE):
            done = True
            rospy.loginfo("y imu")
        if self.episode_step >= self.MAX_EPISODE:
            done = True
            rospy.loginfo("max_episode")

        # rospy.loginfo("Step" + str(self.episode_step) + " : " + str(done))
        return self.getStateY(), self.getStateX(), reward_y, reward_x, done


    def resetWorld(self):
        rospy.wait_for_service('/gazebo/reset_world')
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_world()
        self.episode_step = 0

    
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


    def getInfo(self):
        leg_position = self.getLegPosition()
        imu_data = self.getImuData()
        data = {'lf':leg_position[0],
                'lh':leg_position[1], 
                'rf':leg_position[2], 
                'rh':leg_position[3],
                'x':imu_data[0],
                'y':imu_data[1],
                'z':imu_data[2]}
        return data

    
    def getStateY(self):
        imu_data = self.getImuData()
        data = [self.leg_y, imu_data[1]]
        return data

    def getStateX(self):
        imu_data = self.getImuData()
        data = [self.leg_x, imu_data[0]]
        return data
