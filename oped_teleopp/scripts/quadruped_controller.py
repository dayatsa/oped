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
        self.IMU_MIN_DEGREE = -15*3/2
        self.IMU_MAX_DEGREE = 15*3/2
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
        self.MAX_DEGREE = 68.7549 #57.2958
        self.MOVE_STEP = 3

        self.lf = 0
        self.lh = 0
        self.rf = 0
        self.rh = 0
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


    def addPosition(self, lf, lh, rf, rh):
        self.lf = self.lf + lf*self.MOVE_STEP
        self.lh = self.lh + lh*self.MOVE_STEP
        self.rf = self.rf + rf*self.MOVE_STEP
        self.rh = self.rh + rh*self.MOVE_STEP

        if self.lf > self.MAX_DEGREE:
            self.lf = self.lf - self.MOVE_STEP
        if self.lh > self.MAX_DEGREE:
            self.lh = self.lh - self.MOVE_STEP
        if self.rf > self.MAX_DEGREE:
            self.rf = self.rf - self.MOVE_STEP
        if self.rh > self.MAX_DEGREE:
            self.rh = self.rh - self.MOVE_STEP

        if self.lf < self.MIN_DEGREE:
            self.lf = self.lf + self.MOVE_STEP
        if self.lh < self.MIN_DEGREE:
            self.lh = self.lh + self.MOVE_STEP
        if self.rf < self.MIN_DEGREE:
            self.rf = self.rf + self.MOVE_STEP
        if self.rh < self.MIN_DEGREE:
            self.rh = self.rh + self.MOVE_STEP              

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
        self.lf = 0
        self.lh = 0
        self.rf = 0
        self.rh = 0
        self.setPosition(0,0,0,0)


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
        self.x = 0
        self.y = 0
        self.z = 0.8
        self.MODEL_URDF = '/home/dayatsa/model_editor_models/oped/src/oped/oped_description/urdf/oped.urdf'
        self.ACTION_N = 16
        self.STATE_SPACE = 7
        self.MAX_EPISODE = 200
        self.episode_step = 0
        Leg.__init__(self)
        MyImu.__init__(self)


    def __str__(self):
        return str(self.x + ", " + self.y + ", " + self.z)


    def step(self, choice):
        '''
        Gives us 16 total movement options.
        '''
        self.episode_step += 1

        if choice == 0:
            self.addPosition(0,0,0,1)
        elif choice == 1:
            self.addPosition(0,0,1,0)
        elif choice == 2:
            self.addPosition(0,1,0,0)
        elif choice == 3:
            self.addPosition(1,0,0,0)
            
        elif choice == 4:
            self.addPosition(0,0,0,-1)
        elif choice == 5:
            self.addPosition(0,0,-1,0)
        elif choice == 6:
            self.addPosition(0,-1,0,0)
        elif choice == 7:
            self.addPosition(-1,0,0,0)

        elif choice == 8:
            self.addPosition(0,0,1,1)
        elif choice == 9:
            self.addPosition(0,1,1,0)
        elif choice == 10:
            self.addPosition(1,1,0,0)
        elif choice == 11:
            self.addPosition(1,0,0,1)
            
        elif choice == 12:
            self.addPosition(0,0,-1,-1)
        elif choice == 13:
            self.addPosition(0,-1,-1,0)
        elif choice == 14:
            self.addPosition(-1,-1,0,0)
        elif choice == 15:
            self.addPosition(-1,0,0,-1)

        new_state = self.getState()
        x = new_state[4]
        y = new_state[5]

        reward = 0
        if x > -self.LIMIT_UPRIGHT and x < self.LIMIT_UPRIGHT:
            reward += 1
        if y > -self.LIMIT_UPRIGHT and y < self.LIMIT_UPRIGHT:
            reward += 1

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
        return new_state, reward, done


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

    
    def getState(self):
        leg_position = self.getLegPosition()
        imu_data = self.getImuData()
        data = [leg_position[0], leg_position[1], leg_position[2], leg_position[3], imu_data[0], imu_data[1], imu_data[2]]
        return data
