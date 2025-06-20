import matplotlib.pyplot as plt

def visualize(true_y, pred_y, odefunc, itr):
	# ax_traj.cla()

	plt.clf()

	for i in range(true_x.shape[0]):
		plt.subplot( 3, 4, 1+i )


		plt.plot(true_x[i], true_y.cpu().numpy()[:, i, 0, 0], 'g-', label='ground truth')
		plt.plot(true_x[i], pred_y.cpu().numpy()[:, i, 0, 0], 'b--', label='Prediction')
 
	plt.legend()

	fig.tight_layout()
	plt.title(itr)
	plt.draw()
	plt.pause(0.001)
