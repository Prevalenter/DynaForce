
import sys
import logging
import math
# import gym
# from gym import spaces
# from gym.utils import seeding
import numpy as np
import pybullet as p2
from pybullet_utils import bullet_client as bc
import warnings

import matplotlib.pyplot as plt

import time
from scipy.spatial.transform import Rotation as R

import pybullet_data

if __name__ == '__main__':

    theta = np.arange(0, 6*np.pi, 6*np.pi/1000)
    # print(theta)

    y = np.sin(theta)*0.4
    z = np.cos(theta)*0.4+0.6
    x = np.ones(1000)*0.8

    data = np.array([x, y, z]).transpose((1, 0))


    taget_xyz = np.array([0.8, -0.2, 0.8])

    p = bc.BulletClient(connection_mode=p2.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")

    p.setGravity(0, 0, -10)

    # robot_urdf_path = 'drop/drop-a6-V2.urdf'
    # robotId = p.loadURDF(robot_urdf_path, [0,0,0], useFixedBase=1, globalScaling=2)
    hand_urdf_path = '../../isaac/assets/hand2hand/GX11promax/urdf/GX11promax.urdf'
    robotId = p.loadURDF(hand_urdf_path, [0,0,0.3], 
                         baseOrientation=p.getQuaternionFromEuler([np.pi/2, 0, 0]),
                         useFixedBase=1, globalScaling=2)

    print('getDynamicsInfo')
    for i in range(6):
        p.changeDynamics(robotId, i, lateralFriction=0.0)
        dynamic_info = p.getDynamicsInfo(robotId, i)
        print(dynamic_info[1], dynamic_info[6], dynamic_info[7])

    print(p.getNumJoints(robotId))

    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-125,
                         cameraPitch=-145, cameraTargetPosition=[0, 0, 0])

    zero_list = [0]*6
    r_k = np.array([0]*6)
    delta_t = 1/240
    M_theta_prev = None
    theta_prev = None
    theta_dot_prev = None

    K = np.diag([1e6]*6)
    print(K)
    r = R.from_rotvec(np.array([180, 0, 0])*np.pi/180).as_quat()
    sensor = []
    contact = []
    force_dynamic = []

    coriolis_list = []
    mass_matrix_list= []
    gravity_list = []
    joint_p_list = []
    tau_list = []

    r_ka1_list = []
    for t in range(int(1e5)):
        if t%50==0: print(t)

        sensor_t = []
        for joint_idx in range(6):
            joint_state = p.getJointState(robotId, joint_idx)
            # get joint position, velocity and torque
            sensor_t.append([joint_state[0], joint_state[1], joint_state[3]])
        sensor.append(sensor_t.copy())
        p.stepSimulation()
        time.sleep(delta_t)
