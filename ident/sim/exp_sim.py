
import numpy as np
import matplotlib.pyplot as plt
from pybullet_utils import bullet_client as bc
import pybullet as p2
import sys
sys.path.append('../..')

import pybullet_data
import time

from utils.identification.gene_robot import get_robot
from utils.identification.support_funcs.regr_data_proc import gen_regr_matrices

from env import GX11PMEnv


def LeastSquares(W, tau):
	return np.linalg.pinv(W) @ tau



class SensorlessEstimator:
	def __init__(self, rbt, q, qd, qdd, tau):
		pass


if __name__ == '__main__':
	# 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13
	result_map = [
		[1, 2, 3],
		[5, 6, 7, 8],
		[10, 11, 12, 13]
	]


	def get_joint_mask(finger_name):
		joint_mask = np.zeros(11)
		if finger_name == 'finger1':
			joint_mask = np.zeros(11)
			joint_mask[:3] = 1
		elif finger_name == 'finger2':
			joint_mask = np.zeros(11)
			joint_mask[3:7] = 1
		elif finger_name == 'finger3':
			joint_mask[7:11] = 1
		else:
			raise ValueError('finger name error')
		joint_mask = joint_mask.astype(bool)
		return joint_mask

	from utils.identification.core import IdentFingaer
	finger_list = []
	for finger_idx in range(3):
		finger_mask = get_joint_mask(f"finger{finger_idx+1}")
		finger = IdentFingaer(f'../../data/model/gx11pm_finger{finger_idx+1}.pkl', joint_mask=finger_mask)
  
		# load data
		data = np.load(f'sensor_{finger_idx}.npy').reshape((-1, 11, 3))
		print(data.shape)

		q = data[:, :, 0]
		qd = data[:, :, 1]
		tau = data[:, :, 2]

		qdd = (qd[1:]-qd[:-1])*240

		q = q[200+1:]
		qd = qd[200+1:]
		tau = tau[200+1:]
		qdd = qdd[200:]

		print(q.shape, qd.shape, qdd.shape, tau.shape)

		finger.ident_params(q, qd, qdd, tau)
		finger_list.append(finger)

	link_jacco_list = [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]
	link_vis_list = [0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 13]
	force_applied = [0, 10, 0]

	random_idx = np.random.randint(11)
	random_idx = 10
	link_force_applied = link_vis_list[random_idx]
	link_jacobian = link_jacco_list[random_idx]
 
 
	if link_jacobian in [6, 11]:
		pos_appied = [0.06, 0, 0]
	else:
		pos_appied = [0, 0, 0]

    
	sim = GX11PMEnv(False)
	joint_pos_initial = [0]*11
	joint_pos_initial[0] = np.pi/4
	sim.set_joint_state(joint_pos_initial)
 
	for t in range(int(400)):
		joint_positions, joint_velocities, joint_accelerations, joint_torques = sim.step()
		# print(joint_positions[-3:])
		if t>10:
			sim.applyExternalForce(link_idx=link_force_applied, force=force_applied, is_vis_link=True, pos=pos_appied)

		sim.draw_link_axis(link_force_applied) # 3, 8

		print('-'*40)
		for finger_idx, finger in enumerate(finger_list):
			
			pred_tau = finger.pred_torque(joint_positions[None], joint_velocities[None], joint_accelerations[None])

			res_tau = joint_torques[finger.joint_mask]-pred_tau
			print(finger_idx, res_tau)
			if (np.abs(res_tau)>0.05).sum()>0:
				joint_idx_forced = np.where(np.abs(res_tau)>0.05)[1]
				collision_contact_link = np.max(joint_idx_forced)

				print('ground truth', link_jacobian, result_map[finger_idx][collision_contact_link])
				print("collision_contact_link: ", finger_idx, collision_contact_link)
    
				if t>25:
					breakpoint()

			else:
				continue
	
			jacobian_np = sim.get_jaccobian(joint_positions, link_idx=link_jacobian, pos=pos_appied)

			jacobian_np[np.abs(jacobian_np)<0.01] = 0
			force_pred = np.linalg.pinv(jacobian_np.round(2).T)[:, finger.joint_mask]@res_tau.T

			print('force_pred', t, finger_idx, force_pred.T, np.linalg.norm(force_pred))
  
	breakpoint()

 






  

