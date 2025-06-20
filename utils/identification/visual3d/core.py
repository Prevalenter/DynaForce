import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Slider, Button, RadioButtons
import sympy
from functools import partial

from cubic import plotCubeAt

symbol_q_list = sympy.symbols('q1 q2 q3 q4 q5 q6 q7')
initial_zero = np.array([0, 0, 0])
class RobotVisual(object):
	def __init__(self, robotdef, traj_path, joint_q, len_axis=0.1):
		self.dof = robotdef.dof
		self.sym_kine = robotdef.geo.T
		self.joint_traj = None
		self.xyz_traj_draw = None
		self.joint_q = joint_q
		self.axis_points = np.array([[0, 0, 0, 1], [len_axis, 0, 0, 1],
		                        [0, len_axis, 0, 1], [0, 0, len_axis, 1]]).T
		self.fig = plt.figure(figsize=plt.figaspect(0.5))
		self.ax = plt.subplot(1, 2, 1, projection='3d')

		self.update()

		joint_traj = np.loadtxt(traj_path,delimiter=' ')
		self.joint_traj = joint_traj
		# print(joint_traj.shape)

		resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
		button = Button(resetax, 'Reset', hovercolor='0.975')
		def reset(event):
			print('On button')
			self.loads_joint_traj(self.joint_traj)
			self.update()

			# for i in range(traj.shape[0]//100):
			# 	print(i)
			# 	self.set_joint_p(traj[i*100])
			# 	# self.ax.set_title(i*100/10000)
			# 	plt.pause(0.001)

		button.on_clicked(reset)

		axcolor = 'lightgoldenrodyellow'
		joint_traj_bar = plt.axes([0.7, 0.7 + 0.07, 0.2, 0.03], facecolor=axcolor)
		joint_traj_slider = Slider(joint_traj_bar, 'traj', 0, self.joint_traj.shape[0]-1, valinit=0)
		joint_traj_slider.on_changed(self.on_traj_slider)
		# left, bottom, width, height
		joint_bar_list = []
		self.slider_list = []
		for i in range(self.dof):
			joint_bar = plt.axes([0.7, 0.7 - 0.07 * i, 0.2, 0.03], facecolor=axcolor)
			joint_bar_list.append(joint_bar)
			self.slider_list.append(Slider(joint_bar, f'Joint {i + 1}', -3, 3, valinit=joint_q[i]))
			self.slider_list[-1].on_changed(partial(self.on_slider, i))

		self.set_joint_p([0] * self.dof)

		plt.show()

	def forward_kine(self, q):
		rst = []
		for i in range(self.dof):
			rst_i = self.sym_kine[i].subs({
				symbol_q_list[i]:q[i] for i in range(self.dof)
			})
			rst.append(np.array(rst_i))
		return np.array(rst)

	def on_traj_slider(self, val):
		# self.joint_q = self.joint_traj[int(val)]
		# print(self.joint_traj[int(val)].shape)
		self.set_joint_p(self.joint_traj[int(val)])
		self.update()

	def on_slider(self, joint_idx, val):
		self.joint_q[joint_idx] = val
		self.update()

	def update(self):
		self.ax.clear()

		rst = self.forward_kine(self.joint_q)
		jonit_position = rst[:, :3, -1]
		jonit_position = np.concatenate([initial_zero[None], jonit_position])

		self.ax.plot(*jonit_position.T, c='k')
		axis_cur = np.dot(rst[-1, :3], self.axis_points)
		c_list = ['r', 'g', 'b']
		for i in range(3):
			self.ax.plot(*axis_cur[:, [0, i+1]], c=c_list[i])

		if self.xyz_traj_draw is not None:
			self.ax.plot(*self.xyz_traj_draw.T, 'g--')

			mask = self.xyz_traj_draw[:, 2] < 0.1
			print(self.xyz_traj_draw.shape, mask.shape)
			self.ax.plot(*self.xyz_traj_draw[mask].T, 'r')

		plotCubeAt(pos=(-0.6, -0.6, 0.1), size=(1.2, 1.2, 1.3),
		           alpha=0.2, ax=self.ax, color='green')

		self.ax.set_xlim(-0.8, 0.8)
		self.ax.set_ylim(-0.8, 0.8)
		self.ax.set_zlim(-0.3, 1.3)
		self.ax.set_xlabel('X');self.ax.set_ylabel('Y');self.ax.set_zlabel('Z');
		self.fig.canvas.draw_idle()

	def set_joint_p(self, joint_p):
		print('set_joint_p', joint_p)
		self.joint_q = joint_p
		for i in range(self.dof): self.slider_list[i].set_val(self.joint_q[i])
		self.update()

	def loads_joint_traj(self, joint_traj, num_sample=101):
		self.joint_traj = joint_traj

		mask = np.linspace(0, self.joint_traj.shape[0], num_sample).astype(int)[:-1]
		joint_traj_sampled = self.joint_traj[mask]

		self.traj_draw = []
		# symbol_q_list[:,:3, -1]
		for i in range(joint_traj_sampled.shape[0]):
			print(i, joint_traj_sampled[i].shape)
			rst_i = self.sym_kine[-1].subs({
				symbol_q_list[j]: joint_traj_sampled[i, j] for j in range(self.dof)
			})
			self.traj_draw.append(rst_i)
		self.xyz_traj_draw = np.array(self.traj_draw)[:,:3, -1]
		print(self.xyz_traj_draw.shape)

if __name__ =="__main__":
	import sys
	sys.path.append('../..')
	from utils import process_data, gene_robot

	rbt = gene_robot.get_robot('../../data/estun/estun_model_mdh.pkl', )
	# opt_traj_1

	# WAM7
	# rbt = gene_robot.get_robot('../../data/estun/WAM7.pkl', )

	print(dir(rbt), rbt.dof)

	# rbt_vis = RobotVisual(rbt, traj_path='../../notebook/traj/traj_27_93')
	# rbt_vis = RobotVisual(rbt, traj_path='../../notebook/opt_traj_1.txt', joint_q=[0]*rbt.dof)
	rbt_vis = RobotVisual(rbt, traj_path='../../traj_gen/traj/opt_traj_0_88.87999725341797.txt', joint_q=[0]*rbt.dof)
	# rbt_vis.set_joint_p([0.5]*6)
