import torch.nn as nn
import torch

class ODEFunc(nn.Module):
	def __init__(self):
		super(ODEFunc, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(2, 128),
			nn.Sigmoid(),
			nn.Linear(128, 128),
			nn.Sigmoid(),
			nn.Linear(128, 50),
			nn.Sigmoid(),
			nn.Linear(50, 2),
		)

		for m in self.net.modules():
			if isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, mean=0, std=0.1)
				nn.init.constant_(m.bias, val=0)

	def forward(self, t_batch, y):
		return self.net(y)



class ODEFuncV(nn.Module):
	def __init__(self, v_interpolation, device):
		super(ODEFuncV, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(3, 128),
			nn.Sigmoid(),
			nn.Linear(128, 128),
			nn.Sigmoid(),
			nn.Linear(128, 50),
			nn.Sigmoid(),
			nn.Linear(50, 2),
		)

		self.v_interpolation = v_interpolation
		self.device = device

		for m in self.net.modules():
			if isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, mean=0, std=0.1)
				nn.init.constant_(m.bias, val=0)

	def forward(self, t_cur, y):
		# print('1', t_cur)
		t_cur = torch.clip(t_cur, 0, 0.084)
		# print('2', t_cur)
		v_cur = torch.tensor(self.v_interpolation(t_cur.detach().numpy())[:, None, None]).to(self.device).float()
		# print('forward')
		y_ext = torch.cat([y, v_cur], axis=2)

		return self.net(y_ext)
