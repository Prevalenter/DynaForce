
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

import scipy.signal as signal
import pybullet_data

if __name__ == '__main__':

    finger_idx = 0
    
    traj = np.load(f'../../data/ident/pos_list_{finger_idx}_0.npy')*(np.pi/180)

    window_size = 100

    for j in range(11):
        traj[:, j] = signal.savgol_filter(traj[:, j], window_size, 3)
    
    joint_mask = np.zeros(11)
    if finger_idx == 0:
        joint_mask[:3] = 1
        finger_end_link = 3
    elif finger_idx == 1:
        joint_mask[3:7] = 1
        finger_end_link = 8
    elif finger_idx == 2:
        joint_mask[7:11] = 1
        finger_end_link = 13
    joint_mask = joint_mask.astype(bool)
    num_finger_joints = joint_mask.sum()

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


    
    delta_t = 1/240


    sensor = []
    contact = []
    force_dynamic = []

    coriolis_list = []
    mass_matrix_list= []
    gravity_list = []
    joint_p_list = []
    tau_list = []

    r_ka1_list = []
    
    
    for t in range(traj.shape[0]):
        if t%50==0: print(t)
        sensor_t = []
        for j in range(11):
            p.setJointMotorControl2(bodyUniqueId=robotId, jointIndex=j,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=traj[t, j],
                                    force=1)

        p.stepSimulation()
        time.sleep(delta_t)
        
        
        sensor_t = []
        for joint_idx in range(11):
            joint_state = p.getJointState(robotId, joint_idx)
            sensor_t.append([joint_state[0], joint_state[1], joint_state[3]])
        sensor.append(sensor_t.copy())
    # breakpoint()
    sensor = np.concatenate(sensor)
    np.save(f'sensor_{finger_idx}.npy', sensor)
        
        
