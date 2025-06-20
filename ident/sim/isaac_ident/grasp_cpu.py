
import numpy as np
import math

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")


# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = 4 # args.num_threads
sim_params.physx.use_gpu = True #args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

print(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# breakpoint()

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, gymapi.PlaneParams())

# set up the env grid
num_envs = 1
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)


env = gym.create_env(sim, env_lower, env_upper, 1)


asset_root = "../../../../isaac/assets/hand2hand/"
asset_file = "GX11promax/urdf/GX11promax.urdf"

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

asset_options.flip_visual_attachments = False
asset_options.fix_base_link = True
asset_options.collapse_fixed_joints = False
asset_options.disable_gravity = False
# asset_options.thickness = 0.001
# asset_options.angular_damping = 0.01

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
cartpole_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# initial root pose for cartpole actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 1.0, 0.0)
initial_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 0.5*np.pi) *\
    gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.0 * np.pi) *\
    gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.0 * np.pi)


hand = gym.create_actor(env, cartpole_asset, initial_pose, 'gx11promax', -1, 1)
# Configure DOF properties
props = gym.get_actor_dof_properties(env, hand)

hand_dofs = gym.get_asset_dof_count(cartpole_asset)
hand_dof_props = gym.get_asset_dof_properties(cartpole_asset)
for i in range(hand_dofs):
    hand_dof_props['effort'][i] = 5
    hand_dof_props['stiffness'][i] = 3
    hand_dof_props['damping'][i] = 0.1
    hand_dof_props['friction'][i] = 0.01
    hand_dof_props['armature'][i] = 0.001

gym.set_actor_dof_properties(env, hand, hand_dof_props)
props = gym.get_actor_dof_properties(env, hand)
print( props )



# add bottle
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
bottle_asset_file = "bottle/bottle.urdf"
bottle_asset_file = gym.load_asset(sim, asset_root, bottle_asset_file, asset_options)

box_pose = gymapi.Transform()

box_pose.p.x = 0.1 # table_pose.p.x + np.random.uniform(-0.2, 0.1)
box_pose.p.y = 1.12 # table_pose.p.y + np.random.uniform(-0.3, 0.3)
box_pose.p.z = 0.0
box_pose.r = gymapi.Quat.from_euler_zyx(np.pi/2, 0, np.pi/2)
box_handle = gym.create_actor(env, bottle_asset_file, box_pose, "bottle", 0, 0)
gym.set_actor_scale(env, box_handle, 0.04)
color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))


finger_idx = 0

# load the traj
import scipy.signal as signal
# load traj
# finger_idx = 2
traj = np.load(f'../../../data/ident/pos_list_{finger_idx}_0.npy')*(np.pi/180)
window_size = 100

for j in range(11):
    traj[:, j] = signal.savgol_filter(traj[:, j], window_size, 3)
print(traj.shape)

traj[:, 1] += 0.8

# Look at the first env
cam_pos = gymapi.Vec3(0.6, 1.5, 0)
cam_target = gymapi.Vec3(0, 1, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

gym.enable_actor_dof_force_sensors(env, hand)

cur_pos = []
cur_vel = []
cur_torque = []

dof_pos_initial = [1.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
idx = 0
# Simulate
while not gym.query_viewer_has_closed(viewer):
    idx += 1
    
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    
    if idx>30:
        dof_pos_initial[1] = 1.3
        
        
    
    # gym.set_actor_dof_position_targets(env, hand, dof_pos_initial)
    
    dof_state = gym.get_actor_dof_states(env, hand, gymapi.STATE_ALL)
    
    cur_pos_i = dof_state['pos'].copy()
    cur_vel_i = dof_state['vel'].copy()
    cur_torque_i = gym.get_actor_dof_forces(env, hand)
    
    print(cur_torque_i[3:7])
    
    cur_pos.append(cur_pos_i)
    cur_vel.append(cur_vel_i)
    cur_torque.append(cur_torque_i)
    

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    gym.sync_frame_time(sim)

cur_pos = np.array(cur_pos)
cur_vel = np.array(cur_vel)
cur_torque = np.array(cur_torque)
print(cur_pos.shape, cur_vel.shape, cur_torque.shape)

print('Done') 

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

exit()

import sys
sys.path.append('../../../')
from utils.identification.core import IdentFingaer

pos_cur = cur_pos
vel = cur_vel
force = cur_torque
dt = 1/60

pos_cur = pos_cur[1:]
acc = (vel[1:] - vel[:-1])/dt
vel = vel[1:]
force = force[1:]

pos_cur = pos_cur[10:-10]
acc = acc[10:-10]
vel = vel[10:-10]
force = force[10:-10]

force = force*100
print(pos_cur.shape, acc.shape, vel.shape, force.shape)


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


finger_i = IdentFingaer(f'../../../data/model/gx11pm_finger{finger_idx+1}.pkl', joint_mask=joint_mask)

print(pos_cur.shape, acc.shape, vel.shape, force.shape)
# breakpoint()

finger_i.ident_params(pos_cur, vel, acc, force)


import matplotlib.pyplot as plt
pred = finger_i.pred_torque(pos_cur, vel, acc)



for joint_idx in range(3):
    plt.subplot(3, 1, joint_idx+1)
    plt.plot(force[:, joint_idx], label='force')
    plt.plot(pred[:, joint_idx].astype(np.float32), label='pred')
    
    plt.legend()
plt.show()


