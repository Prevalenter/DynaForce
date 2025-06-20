import pybullet as p
import pybullet_data
import time
import numpy as np

import argparse

import scipy.signal as signal

# 连接到 PyBullet 物理引擎
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# finger_idx = 2
# hand_type = 'gx11'  # 可根据需要修改

parser = argparse.ArgumentParser(description='Pick and Place Simulation')
# 添加 hand_type 参数，默认值为 gx11
parser.add_argument('--hand_type', type=str, default='gx11', help='Type of hand')
# 添加 finger_idx 参数，默认值为 2
parser.add_argument('--finger_idx', type=int, default=0, help='Index of finger')
# 解析命令行参数
args = parser.parse_args()

# 使用解析后的参数
hand_type = args.hand_type
finger_idx = args.finger_idx

traj = np.load(f'../../../data/ident/pos_list_{finger_idx}_0.npy')*(np.pi/180)

window_size = 100

for j in range(11):
    traj[:, j] = signal.savgol_filter(traj[:, j], window_size, 3)

if hand_type == 'gx11super':
    hand_urdf_path = "../../../isaac/assets/hand2hand/GX11promax/urdf/GX11promax.urdf"
elif hand_type == 'gx11':
    hand_urdf_path = "../../../isaac/assets/hand2hand/GX11pro_plus3/urdf/GX11pro_plus3.urdf"
elif hand_type == 'gx11ball':
    hand_urdf_path = "../../../isaac/assets/hand2hand/GX11pro_plus3/urdf/GX11pro_plus3_ball.urdf"
else:
    raise NotImplementedError

# 加载 URDF 文件
hand_id = p.loadURDF(hand_urdf_path, [0, 0, 0], useFixedBase=1, baseOrientation=p.getQuaternionFromEuler([np.pi, -np.pi/2, np.pi/2]))

# 获取非固定关节数量
num_joints = p.getNumJoints(hand_id)
non_fixed_joints = []
for joint_index in range(num_joints):
    joint_info = p.getJointInfo(hand_id, joint_index)
    joint_type = joint_info[2]
    if joint_type != p.JOINT_FIXED:
        non_fixed_joints.append(joint_index)

p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=-80,
                    cameraPitch=-145, cameraTargetPosition=[0, 0, 0.1])

sensor = []
# 主循环，通过滑块条控制关节
# while True:
for traj_idx in range(traj.shape[0]):
    for i, joint_index in enumerate(non_fixed_joints):
        p.setJointMotorControl2(hand_id, joint_index, p.POSITION_CONTROL, targetPosition=traj[traj_idx, i], force=10)
    p.stepSimulation()
    time.sleep(1./240.)

    
    sensor_t = []
    for joint_idx in range(11):
        joint_state = p.getJointState(hand_id, joint_idx)
        sensor_t.append([joint_state[0], joint_state[1], joint_state[3]])
    sensor.append(sensor_t.copy())
    
# breakpoint()
sensor = np.concatenate(sensor)
np.save(f'data/traj/sensor_{hand_type}_{finger_idx}.npy', sensor)


