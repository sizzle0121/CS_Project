import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class QNet(nn.Module):
	def __init__(self, in_planes, opt):
		super(QNet,self).__init__()
		self.conv1 = nn.Conv2d(in_planes, 32, kernel_size = 8, stride = 4, bias = False)	#84 -> 20
		self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2, bias = False)			#20 -> 9
		self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, bias = False)			# 9 -> 7
		self.fc1 = nn.Linear(64*7*7, 512)
		self.fc2 = nn.Linear(512, opt)
		#self.init_weight()

	def init_weight(self):
		nn.init.kaiming_normal_(self.conv1.weight.data, mode = 'fan_in', nonlinearity = 'relu')
		nn.init.kaiming_normal_(self.conv2.weight.data, mode = 'fan_in', nonlinearity = 'relu')
		nn.init.kaiming_normal_(self.conv3.weight.data, mode = 'fan_in', nonlinearity = 'relu')
		nn.init.kaiming_normal_(self.fc1.weight.data, mode = 'fan_in', nonlinearity = 'relu')
		nn.init.kaiming_normal_(self.fc2.weight.data, mode = 'fan_in', nonlinearity = 'relu')

	def forward(self, x):
		x = F.leaky_relu(self.conv1(x), 0.01, inplace = True)
		x = F.leaky_relu(self.conv2(x), 0.01, inplace = True)
		x = F.leaky_relu(self.conv3(x), 0.01, inplace = True)
		x = x.view(x.size(0), -1)
		x = F.leaky_relu(self.fc1(x), 0.01, inplace = True)
		x = self.fc2(x)
		return x


