import numpy as np
import matplotlib.pyplot as plt
from pybullet_utils import bullet_client as bc
import pybullet as p2
import sys
sys.path.append('../..')

import pybullet_data
import time
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton, QMessageBox
import threading
from PyQt5.QtCore import pyqtSignal, QObject

from utils.identification.gene_robot import get_robot
from utils.identification.support_funcs.regr_data_proc import gen_regr_matrices

def LeastSquares(W, tau):
	return np.linalg.pinv(W) @ tau


def getMotorJointStates(p, robot):
	joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
	joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
	joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
	joint_positions = [state[0] for state in joint_states]
	joint_velocities = [state[1] for state in joint_states]
	joint_torques = [state[3] for state in joint_states]
	return joint_positions, joint_velocities, joint_torques

class Sim:
	def __init__(self, headless=False):
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
		# breakpoint()

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
		hand_urdf_path = '/home/gx4070/data/lx/DynaForce/data/GX11promax/urdf/GX11promax.urdf'
		robotId = self.p.loadURDF(hand_urdf_path, [0,0,0.3], 
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


	def get_jaccobian(self, joint_pos, link_idx, pos=[0, 0, 0]):
		
		# com_trn
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
  

class SensorlessEstimator:
	def __init__(self, rbt, q, qd, qdd, tau):
		pass

def get_finger_info(finger_name):
	if finger_name == 'finger1':
		finger_idx = 0
		joint_mask = np.zeros(11)
		joint_mask[:3] = 1
		dof = 3
	elif finger_name == 'finger2':
		finger_idx = 1
		joint_mask = np.zeros(11)
		joint_mask[3:7] = 1
		dof = 4
	elif finger_name == 'finger3':
		finger_idx = 2
		joint_mask = np.zeros(11)
		joint_mask[7:11] = 1
		dof = 4
	else:
		raise ValueError('finger name error')
	joint_mask = joint_mask.astype(bool)
	return finger_idx, joint_mask, dof

if __name__ == '__main__':
	hand_config = {
		'finger1': {
			1: (0, [0, 1, 0]), 
			2: (1, [0, 1, 1]), 
			3: (3, [1, 1, 0]),
		},
		'finger2': {
			5: (4, [0, 1, 0]),
			6: (5, [0, 1, 1]),
			7: (6, [0, 1, 1]),
			8: (8, [1, 1, 0]), 
		},
		'finger3': {
			10: (9, [0, 1, 0]),
			11: (10, [0, 1, 1]),
			12: (11, [0, 1, 1]), 
			13: (13, [1, 1, 0]),
		}
	}

	class SelectionState(QObject):
		def __init__(self, hand_config):
			super().__init__()
			self.hand_config = hand_config
			self.finger = 'finger1'
			self.link = 1
			self.lock = threading.Lock()
		def get(self):
			with self.lock:
				return self.finger, self.link
		def set(self, finger, link):
			with self.lock:
				self.finger = finger
				self.link = link

	selection_state = SelectionState(hand_config)

	class SelectionDialog(QDialog):
		def __init__(self, hand_config, selection_state):
			super().__init__()
			self.hand_config = hand_config
			self.selection_state = selection_state
			self.init_ui()
		def init_ui(self):
			self.setWindowTitle('Select Finger and Link (Live)')
			self.setFixedSize(400, 200)  # Make the window larger
			layout = QVBoxLayout()
			label_style = 'font-size: 18px;'
			combo_style = 'font-size: 16px; min-height: 32px;'
			label_finger = QLabel('Select Finger:')
			label_finger.setStyleSheet(label_style)
			layout.addWidget(label_finger)
			self.finger_combo = QComboBox()
			self.finger_combo.addItems(self.hand_config.keys())
			self.finger_combo.setStyleSheet(combo_style)
			self.finger_combo.currentTextChanged.connect(self.update_links)
			layout.addWidget(self.finger_combo)
			label_link = QLabel('Select Link:')
			label_link.setStyleSheet(label_style)
			layout.addWidget(label_link)
			self.link_combo = QComboBox()
			self.link_combo.setStyleSheet(combo_style)
			layout.addWidget(self.link_combo)
			self.update_links(self.finger_combo.currentText())
			self.finger_combo.setCurrentText(self.selection_state.finger)
			self.link_combo.setCurrentText(str(self.selection_state.link))
			self.finger_combo.currentTextChanged.connect(self.on_selection_change)
			self.link_combo.currentTextChanged.connect(self.on_selection_change)
			self.setLayout(layout)
		def update_links(self, finger):
			self.link_combo.clear()
			links = list(self.hand_config[finger].keys())
			self.link_combo.addItems([str(l) for l in links])
			if links:
				self.link_combo.setCurrentText(str(links[0]))
		def on_selection_change(self):
			finger = self.finger_combo.currentText()
			link = self.link_combo.currentText()
			if link:
				self.selection_state.set(finger, int(link))

	# Load and identify all three robot models and their data at startup
	FINGER_INFO = {
		'finger1': {'idx': 0, 'mask': (slice(0, 3), 3)},
		'finger2': {'idx': 1, 'mask': (slice(3, 7), 4)},
		'finger3': {'idx': 2, 'mask': (slice(7, 11), 4)},
	}
	rbt_dict = {}
	data_dict = {}
	ident_dict = {}
	for fname, info in FINGER_INFO.items():
		idx = info['idx']
		mask_slice, dof = info['mask']
		# Load robot model
		rbt = get_robot(f'../../data/model/gx11pm_finger{idx+1}.pkl')
		rbt_dict[fname] = rbt
		# Load data
		data = np.load(f'sensor_{idx}.npy').reshape((-1, 11, 3))
		q = data[:, :, 0][:, mask_slice]
		qd = data[:, :, 1][:, mask_slice]
		tau = data[:, :, 2][:, mask_slice]
		qdd = (qd[1:]-qd[:-1])*240
		q = q[200+1:]
		qd = qd[200+1:]
		tau = tau[200+1:]
		qdd = qdd[200:]
		data_dict[fname] = {
			'idx': idx,
			'mask_slice': mask_slice,
			'dof': dof,
			'q': q,
			'qd': qd,
			'tau': tau,
			'qdd': qdd
		}
		# Identification
		W, omega, Q1, R1, rho1 = gen_regr_matrices(rbt, q, qd, qdd, tau)
		beta_ols = LeastSquares(W, omega)
		ident_dict[fname] = {
			'W': W,
			'omega': omega,
			'Q1': Q1,
			'R1': R1,
			'rho1': rho1,
			'beta_ols': beta_ols
		}

	def simulation_thread_func():
		finger_name, link_jacobian = selection_state.get()
		print(f'Initial selection: finger: {finger_name}, link_idx: {link_jacobian}')
		link_force_applied, available_force_axes = hand_config[finger_name][link_jacobian]
		sim = Sim(False)
		joint_pos_initial = [0]*11
		joint_pos_initial[0] = np.pi/4
		sim.set_joint_state(joint_pos_initial)
		for t in range(int(40000)):
			# Always get the latest selection for force application
			finger_name, link_jacobian = selection_state.get()
			link_force_applied, available_force_axes = hand_config[finger_name][link_jacobian]
			if link_jacobian in [6, 11]:
				pos_appied = [0.06, 0, 0]
			else:
				pos_appied = [0, 0, 0]
			joint_positions, joint_velocities, joint_accelerations, joint_torques = sim.step()
			# Apply force only to the selected finger
			if t>10:
				sim.applyExternalForce(link_idx=link_force_applied, force=[0, 10, 0], is_vis_link=False, pos=pos_appied)
			sim.draw_link_axis(link_force_applied)
			# Loop over all three robots for collision detection
			for fname in ['finger1', 'finger2', 'finger3']:
				mask_slice = data_dict[fname]['mask_slice']
				dof = data_dict[fname]['dof']
				num_finger_joints = dof
				rbt = rbt_dict[fname]
				beta_ols = ident_dict[fname]['beta_ols']
				# Use only the relevant joints for this finger
				joint_positions_f = np.array(joint_positions)[mask_slice]
				joint_velocities_f = np.array(joint_velocities)[mask_slice]
				joint_accelerations_f = np.array(joint_accelerations)[mask_slice]
				joint_torques_f = np.array(joint_torques)[mask_slice]
				joint_torque_zero = np.zeros_like(joint_torques_f)
				W, omega, Q1, R1, rho1 = gen_regr_matrices(
					rbt,
					joint_positions_f[None],
					joint_velocities_f[None],
					joint_accelerations_f[None],
					joint_torque_zero[None]
				)
				pred_tau = (W @ beta_ols).reshape((-1, num_finger_joints))
				res_tau = joint_torques_f - pred_tau

				if (np.abs(res_tau)>0.05).sum()>0:
					joint_idx_forced = np.where(np.abs(res_tau)>0.05)[1]
					collision_contact_link = np.max(joint_idx_forced)

					link_name_map = {
						'finger1': {
							0: 'link1', 
							1: 'link2', 
							2: 'link3',
						},
						'finger2': {
							0: 'link4',
							1: 'link5',
							2: 'link6',
							3: 'link7',
						},
						'finger3': {
							0: 'link8',
							1: 'link9',
							2: 'link10',
							3: 'link11',
						}
					}
					collision_contact_link_name = link_name_map[fname][collision_contact_link]
     
					print(f"[Collision detection result] {fname}: {collision_contact_link_name}, pred_tau: {pred_tau}")
		print('Simulation finished.')

	# Start the simulation in a background thread
	sim_thread = threading.Thread(target=simulation_thread_func, daemon=True)
	sim_thread.start()

	# Start the PyQt UI in the main thread
	app = QApplication(sys.argv)
	dialog = SelectionDialog(hand_config, selection_state)
	dialog.show()
	app.exec_()
	print('UI closed. Exiting main thread.')


  

