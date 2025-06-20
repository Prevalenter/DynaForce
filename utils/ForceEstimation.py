import sys
sys.path.append('../..')
sys.path.append('../../..')

import numpy as np


from ident.sim.env import GX11PMEnv
from utils.identification.core import IdentFingaer, process_data

class HandForce:
    def __init__(self, hand_urdf_path='/home/gx4070/data/lx/DynaForce/data/GX11promax/urdf/GX11promax.urdf'):
        
        self.urdf_sign = np.array([1, 1, 1,
                            1, -1, -1, -1,
                            -1, -1, 1, 1])[None]
        self.result_map = [
            [0, 1, 2],
            [3, 4, 5, 6],
            [7, 8, 9, 10]
        ]
        
        self.finger_fix_threshold_scale = [np.array([0.03 , 0.06, 0.024]), 
                                    np.array([0.1, 0.05, 0.04, 0.035]), 
                                    np.array([0.1, 0.05, 0.05, 0.024])]
    
        # the jaccobian need the pybullet env
        self.env_pybullet = GX11PMEnv(hand_urdf_path=hand_urdf_path, headless=True)
    
    def load_default_params(self, dir='../../data', algo = 'ols'):
        
        # get the abs path
        import os
        print(os.path.abspath(dir))
        
        self.finger_list = []    
        for joint_idx in [0, 1, 2]:
            print(f'joint_idx: {joint_idx}')

            exp_idx = 2
            
            joint_mask = np.zeros(11)
            if joint_idx == 0:
                joint_mask[:3] = 1
            elif joint_idx == 1:
                joint_mask[3:7] = 1
            elif joint_idx == 2:
                joint_mask[7:11] = 1

            joint_mask = joint_mask.astype(bool)
            

            finger = IdentFingaer(f'{dir}/model/gx11pm_finger{joint_idx+1}.pkl', joint_mask=joint_mask)
            q_raw = np.load(f'{dir}/ident/pos_list_{joint_idx}_{exp_idx}.npy')
            torque_raw = np.load(f'{dir}/ident/torque_list_{joint_idx}_{exp_idx}.npy')
            t_list_raw = np.load(f'{dir}/ident/t_list_{joint_idx}_{exp_idx}.npy')

            # get beta
            q, qd, qdd, torque = process_data(q_raw.copy(), torque_raw.copy(), t_list_raw.copy(), urdf_sign=self.urdf_sign)
            finger.ident_params(q, qd, qdd, torque, algo=algo)
            
            # breakpoint()
            self.finger_list.append(finger)

    def init_fingers(self):
        self.finger_list = []
        for joint_idx in [0, 1, 2]:
            print(f'joint_idx: {joint_idx}')

            joint_mask = np.zeros(11)
            if joint_idx == 0:
                joint_mask[:3] = 1
            elif joint_idx == 1:
                joint_mask[3:7] = 1
            elif joint_idx == 2:
                joint_mask[7:11] = 1

            joint_mask = joint_mask.astype(bool)
            
            finger = IdentFingaer(f'../../data/model/gx11pm_finger{joint_idx+1}.pkl', joint_mask=joint_mask)
            self.finger_list.append(finger)

    def set_finger_params(self, finger_idx, beta):
        self.finger_list[finger_idx].set_params(beta)

    def estimate_force(self, q, qd, qdd, cur, default_joint=None):
        force_pred_list = []
        collision_links = []
        collision_link_names = [None, None, None]
        for finger_idx, finger in enumerate(self.finger_list):
            cur_fingertip_force = np.zeros((3, 1))
            
            torque = cur[finger.joint_mask]

            # pred torque
            pred_torque_i = finger.pred_torque(q[None], qd[None], qdd[None])
            pred_torque_i = np.array(pred_torque_i)
            error = torque[None] - pred_torque_i
            
            collision_joint_idx = np.abs(error)>self.finger_fix_threshold_scale[finger_idx][None]
            if (collision_joint_idx).sum()>0:

                joint_idx_forced = np.where(collision_joint_idx)[1]
                collision_contact_link = np.max(joint_idx_forced)
                
                # only in the fingertip
                cur_collision_link = self.result_map[finger_idx][collision_contact_link]
                # if cur_collision_link in [2, 6, 10]:
                if True:
                    collision_links.append(cur_collision_link)

                    if finger_idx == 0:
                        link_name = "hand_/Link12"
                        jac_link_idx = 3
                    elif finger_idx == 1:
                        link_name = "hand_/Link13"
                        jac_link_idx = 8
                    elif finger_idx == 2:
                        link_name = "hand_/Link14"
                        jac_link_idx = 13
                        
                    # collision_link_names.append(link_name)
                    collision_link_names[finger_idx] = link_name
                
                    # get the jacobian
                    # cur_pos_rad = cur_pos # *(np.pi/180)
                    if default_joint is not None:
                        jacobian_np = self.env_pybullet.get_jaccobian(joint_pos=default_joint, link_idx=jac_link_idx)
                    else:
                        jacobian_np = self.env_pybullet.get_jaccobian(joint_pos=q, link_idx=jac_link_idx)

                    jacobian_np[np.abs(jacobian_np)<0.01] = 0

                    cur_finger_mask = np.abs(error)>self.finger_fix_threshold_scale[finger_idx][None]
                    error[cur_finger_mask==False] = 0
                    res_tau = error


                    force_pred = np.linalg.pinv(jacobian_np.round(2).T)[:, finger.joint_mask]@res_tau.T

                    cur_fingertip_force = force_pred

            force_pred_list.append(cur_fingertip_force)
    
        return collision_links, collision_link_names, force_pred_list

    def estimate_force_without_threshold(self, q, qd, qdd, cur, default_joint=None, jacobian_np_threshold=0.01):
        force_pred_list = []
        for finger_idx, finger in enumerate(self.finger_list):
            cur_fingertip_force = np.zeros((3, 1))
            
            # breakpoint
            torque = cur[finger.joint_mask]

            # pred torque
            pred_torque_i = finger.pred_torque(q[None], qd[None], qdd[None])
            pred_torque_i = np.array(pred_torque_i)
            error = torque[None] - pred_torque_i

            if finger_idx == 0:
                link_name = "hand_/Link12"
                jac_link_idx = 3
            elif finger_idx == 1:
                link_name = "hand_/Link13"
                jac_link_idx = 8
            elif finger_idx == 2:
                link_name = "hand_/Link14"
                jac_link_idx = 13

            if default_joint is not None:
                jacobian_np = self.env_pybullet.get_jaccobian(joint_pos=default_joint, link_idx=jac_link_idx)
            else:
                jacobian_np = self.env_pybullet.get_jaccobian(joint_pos=q, link_idx=jac_link_idx)

            jacobian_np[np.abs(jacobian_np)<jacobian_np_threshold] = 0

            res_tau = error

            force_pred = np.linalg.pinv(jacobian_np.round(2).T)[:, finger.joint_mask]@res_tau.T

            cur_fingertip_force = force_pred

            force_pred_list.append(cur_fingertip_force)
    
        return force_pred_list
    
    def estimate_force_momentum(self, q, qd, qdd, cur, default_joint=None, jacobian_np_threshold=0.01):
        force_pred_list = []
        for finger_idx, finger in enumerate(self.finger_list):
            cur_fingertip_force = np.zeros((3, 1))
            
            # breakpoint
            torque = cur[finger.joint_mask]

            # pred torque
            error = finger.pred_torque_momentum(q[None], qd[None], qdd[None], torque[None])
            # pred_torque_i = np.array(pred_torque_i)
            # error = torque[None] - pred_torque_i

            if finger_idx == 0:
                link_name = "hand_/Link12"
                jac_link_idx = 3
            elif finger_idx == 1:
                link_name = "hand_/Link13"
                jac_link_idx = 8
            elif finger_idx == 2:
                link_name = "hand_/Link14"
                jac_link_idx = 13

            if default_joint is not None:
                jacobian_np = self.env_pybullet.get_jaccobian(joint_pos=default_joint, link_idx=jac_link_idx)
            else:
                jacobian_np = self.env_pybullet.get_jaccobian(joint_pos=q, link_idx=jac_link_idx)

            jacobian_np[np.abs(jacobian_np)<jacobian_np_threshold] = 0

            res_tau = error

            force_pred = np.linalg.pinv(jacobian_np.round(2).T)[:, finger.joint_mask]@res_tau.T

            cur_fingertip_force = force_pred

            force_pred_list.append(cur_fingertip_force)
    
        return force_pred_list
    