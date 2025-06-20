import pybullet as p2
from pybullet_utils import bullet_client as bc
import pybullet_data
import time
import numpy as np



def getMotorJointStates(p, robot):
	joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
	joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
	joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
	joint_positions = [state[0] for state in joint_states]
	joint_velocities = [state[1] for state in joint_states]
	joint_torques = [state[3] for state in joint_states]
	return joint_positions, joint_velocities, joint_torques

class GX11PMEnv:
	def __init__(self, headless=False, hand_urdf_path='../../../isaac/assets/hand2hand/GX11promax/urdf/GX11promax.urdf'):
		self.hand_urdf_path = hand_urdf_path
     
		# to headless
		delta_t = 1/240
		self.delta_t = delta_t

		# p = bc.BulletClient(connection_mode=p2.DIRECT)
		p = bc.BulletClient(connection_mode=p2.DIRECT if headless else p2.GUI )
		self.p = p
		p.setTimeStep(delta_t)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		planeId = p.loadURDF("plane.urdf")

		p.setGravity(0, 0, -10)

		self.create_env()

		p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-125,
								cameraPitch=-145, cameraTargetPosition=[0, 0, 0])

		num_joints = p.getNumJoints(self.robotId)

		print('getDynamicsInfo')
		for i in range(num_joints):
			p.changeDynamics(self.robotId, i, lateralFriction=0.0)
			dynamic_info = p.getDynamicsInfo(self.robotId, i)
			print(dynamic_info[1], dynamic_info[6], dynamic_info[7])
  
		print(p.getNumJoints(self.robotId))

		# print all joint name
		for i in range(num_joints):
			joint_info = p.getJointInfo(self.robotId, i)
			joint_name = joint_info[1].decode('utf-8')  # 第1个元素是关节名称
			print(f"Joint {i} name: {joint_name}")
			print(f'Link name {i}: ', joint_info[12].decode('utf-8'))
 
 
		# get the num joint not fixed
		num_joints = self.p.getNumJoints(self.robotId)
		num_joints_not_fixed = 0
		self.joints_not_fixed = []
		for i in range(num_joints):
			joint_info = self.p.getJointInfo(self.robotId, i)

			if joint_info[3] > -1:
				num_joints_not_fixed += 1
				self.joints_not_fixed.append(i)
		print('num_joints_not_fixed: ', num_joints_not_fixed)

		self.num_joints_not_fixed = num_joints_not_fixed

		self.last_velocity = np.zeros(self.num_joints_not_fixed)
		
	def create_env(self):

		robotId = self.p.loadURDF(self.hand_urdf_path, [0,0,0.3], 
								baseOrientation=self.p.getQuaternionFromEuler([np.pi/2, 0, 0]),
								useFixedBase=1, globalScaling=2)
		self.robotId = robotId
  
	def step(self, action=None):
		# get link name
		for j in range(self.num_joints_not_fixed):
			if action ==None:
				self.p.setJointMotorControl2(bodyUniqueId=self.robotId, 
                                 		jointIndex=self.joints_not_fixed[j],
										controlMode=self.p.POSITION_CONTROL,
										targetPosition=0 if j!=0 else np.pi/4,
										force=200)
			else:
				self.p.setJointMotorControl2(bodyUniqueId=self.robotId, 
                                 		jointIndex=self.joints_not_fixed[j],
										controlMode=self.p.POSITION_CONTROL,
										targetPosition=action[j],
										force=200)

		# if t%50==0: print(t)
		self.p.stepSimulation()
		time.sleep(self.delta_t)
		return self.get_joint_state()

	def set_joint_state(self, joint_pos):
		for j in range(self.num_joints_not_fixed):
			self.p.resetJointState(self.robotId, self.joints_not_fixed[j], joint_pos[j])

	def get_joint_state(self):
		joint_positions, joint_velocities, joint_torques = getMotorJointStates(self.p, self.robotId)

		joint_accelerations = (np.array(joint_velocities) - self.last_velocity) / self.delta_t
  
		self.last_velocity = joint_velocities
		return np.array(joint_positions), np.array(joint_velocities), np.array(joint_accelerations), np.array(joint_torques)

	def applyExternalForce(self, link_idx, force, is_vis_link=False, pos=[0, 0, 0]):
		self.p.applyExternalForce(self.robotId, link_idx, force, pos, self.p.LINK_FRAME)
		if is_vis_link:
			self.p.changeVisualShape(self.robotId, link_idx, rgbaColor=[0, 1, 0, 1])

	def changeVisualShape(self, link_idx, rgbaColor=[1, 0, 0, 1]):
		print(link_idx)
		# breakpoint()
		self.p.changeVisualShape(self.robotId, link_idx, rgbaColor=rgbaColor)

	def get_jaccobian(self, joint_pos, link_idx, pos=[0, 0, 0]):
		# com_trn
		# breakpoint()
		jacobian_matrix = self.p.calculateJacobian(self.robotId, link_idx, pos, 
										list(joint_pos), [0]*self.num_joints_not_fixed, [0]*self.num_joints_not_fixed)
		return np.array(jacobian_matrix[0])

	def draw_link_axis(self, link_idx):
		link_trn, link_rot = self.p.getLinkState(self.robotId, link_idx)[:2]
		# link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result[2]

		link_rot = np.array(link_rot)
 
		# draw the link3 frame
		link_rot_matrix = np.array(self.p.getMatrixFromQuaternion(link_rot)).reshape(-1, 3)
  
		# breakpoint()
		axis_x = link_trn+link_rot_matrix[:, 0]
		axis_y = link_trn+link_rot_matrix[:, 1]
		axis_z = link_trn+link_rot_matrix[:, 2]
  
		self.p.addUserDebugLine(link_trn, axis_x, [1, 0, 0], 3, 0.3)
		self.p.addUserDebugLine(link_trn, axis_y, [0, 1, 0], 3, 0.3)
		self.p.addUserDebugLine(link_trn, axis_z, [0, 0, 1], 3, 0.3)
  


