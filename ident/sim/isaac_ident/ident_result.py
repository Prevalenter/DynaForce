import sys
import numpy as np

sys.path.append('../../../')


if __name__=="__main__":
    from scipy.spatial.transform import Rotation as R
    import matplotlib.pyplot as plt
    
    from utils.identification.core import IdentFingaer
    
    hand_type = 'gx11super'
    finger_idx = 0
    
    dir_save = '../../../data/ident_task/pick_place/traj'
    measure = np.load(f'{dir_save}/measure_{hand_type}_{finger_idx}.npy')

    pos_cur = measure[:, 0, 0, 7:]
    vel = measure[:, 1, 0, 7:]
    force = measure[:, 2, 0, 7:]

    dt = 1/60

    pos_cur = pos_cur[1:]
    acc = (vel[1:] - vel[:-1])/dt
    vel = vel[1:]
    force = force[1:]
    
    pos_cur = pos_cur[10:-10]
    acc = acc[10:-10]
    vel = vel[10:-10]
    force = force[10:-10]
    print(pos_cur.shape, acc.shape, vel.shape, force.shape)
    
    # force[:, 1] *= -1
    # breakpoint()
    
    if finger_idx==0:
        joint_mask=np.array([1, 1, 1,
                            0, 0, 0, 0,
                            0, 0, 0, 0])
    elif finger_idx==1:
        joint_mask=np.array([0, 0, 0,
                            1, 1, 1, 1,
                            0, 0, 0, 0])
    elif finger_idx==2:
        joint_mask=np.array([0, 0, 0,
                            0, 0, 0, 0,
                            1, 1, 1, 1])


    print(measure.shape)
    finger_i = IdentFingaer(f'../../../data/model/gx11pm_finger{finger_idx+1}.pkl', joint_mask=joint_mask)
    finger_i.ident_params(pos_cur, vel, acc, force)

    
    pred = finger_i.pred_torque(pos_cur, vel, acc)
    
    for joint_idx in range(3):
        plt.subplot(3, 1, joint_idx+1)
        plt.plot(force[:, joint_idx], label='force')
        plt.plot(pred[:, joint_idx], label='pred')
        
        plt.legend()
            
        
        
    plt.show()
    
    
    