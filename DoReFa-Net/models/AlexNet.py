import torch.nn as nn

from tools import conv2d_Q_fn, linear_Q_fn, activation_quantize_fn

class AlexNet(nn.Module):
	def __init__(self, wbits=32, abits=32):
		super(AlexNet, self).__init__()
		Conv2d = conv2d_Q_fn(w_bit=wbits)
		Linear = linear_Q_fn(w_bit=wbits)
		self.act_q = activation_quantize_fn(a_bit=abits)
		self.relu = nn.ReLU(inplace=True)

		#first layer not quantized
		self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0, bias=False)
		self.bn1 = nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True)
		self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)
		# ReLU

		#activation_q
		self.conv2 = Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=1, bias=False)
		self.bn2 = nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=True)
		self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2)
		# ReLU

		#activation_q
		self.conv3 = Conv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(384, eps=1e-4, momentum=0.1, affine=True)
		# ReLU

		#activation_q
		self.conv4 = Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
		self.bn4 = nn.BatchNorm2d(384, eps=1e-4, momentum=0.1, affine=True)
		# ReLU

		#activation_q
		self.conv5 = Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
		self.bn5 = nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=True)
		self.mp3 = nn.MaxPool2d(kernel_size=3, stride=2)
		# ReLU

		#reshape

		#activation_q
		self.lin1 = Linear(256*6*6, 4096, bias=False)
		self.bn6 = nn.BatchNorm1d(4096, eps=1e-4, momentum=0.1, affine=True)
		# ReLU

		#self.dp1 = nn.Dropout(0.5)
		#activation_q
		self.lin2 = Linear(4096, 4096, bias=False)
		self.bn7 = nn.BatchNorm1d(4096, eps=1e-4, momentum=0.1, affine=True)
		# ReLU

		#self.dp2 = nn.Dropout(0.5)
		#last layer not quantized
		self.lin3 = nn.Linear(4096, 1000, bias=False)
		self.bn8 = nn.BatchNorm1d(1000, eps=1e-3, momentum=0.1, affine=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.mp1(x)
		x = self.relu(x)

		x = self.act_q(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.mp2(x)
		x = self.relu(x)

		x = self.act_q(x)
		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu(x)

		x = self.act_q(x)
		x = self.conv4(x)
		x = self.bn4(x)
		x = self.relu(x)

		x = self.act_q(x)
		x = self.conv5(x)
		x = self.bn5(x)
		x = self.mp3(x)
		x = self.relu(x)

		x = x.view(x.size(0), -1)

		x = self.act_q(x)
		x = self.lin1(x)
		x = self.bn6(x)
		x = self.relu(x)

		#x = self.dp1(x)
		x = self.act_q(x)
		x = self.lin2(x)
		x = self.bn7(x)
		x = self.relu(x)

		#x = self.dp2(x)
		x = self.lin3(x)
		x = self.bn8(x)

		return x