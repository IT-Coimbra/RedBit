import torch.nn as nn

from tools import conv2d_Q_fn, linear_Q_fn, activation_quantize_fn

class LeNet_5(nn.Module):
	def __init__(self, wbits=32, abits=32):
		super(LeNet_5, self).__init__()
		Conv2d = conv2d_Q_fn(w_bit=wbits)
		Linear = linear_Q_fn(w_bit=wbits)
		self.act_q = activation_quantize_fn(a_bit=abits)
		self.relu=nn.ReLU(inplace=True)
		
		#first layer not quantized
		self.conv1=nn.Conv2d(1, 20, kernel_size=5, stride=1, bias=False)
		self.bn1=nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=False)
		# ReLU
		self.ap1=nn.AvgPool2d(kernel_size=2, stride=2)

		#activation_q
		self.conv2=Conv2d(20, 50, kernel_size=5, stride=1, padding=0, bias=False)
		self.bn2=nn.BatchNorm2d(50, eps=1e-4, momentum=0.1, affine=True)
		# ReLU
		self.ap2=nn.AvgPool2d(kernel_size=2, stride=2)

		#activation_q
		self.l1=Linear(50*4*4, 500, bias=False)
		self.bn3=nn.BatchNorm1d(500, eps=1e-4, momentum=0.1, affine=True)
		# ReLU

		#activation_q
		self.l2=Linear(500, 250, bias=False)
		self.bn4=nn.BatchNorm1d(250, eps=1e-4, momentum=0.1, affine=True)
		# ReLU
		
		#last layer not quantized
		self.l3=nn.Linear(250, 10, bias=False)
		

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.ap1(x)

		x = self.act_q(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)
		x = self.ap2(x)

		x = x.view(x.size(0), -1)

		x = self.act_q(x)
		x = self.l1(x)
		x = self.bn3(x)
		x = self.relu(x)
		
		x = self.act_q(x)
		x = self.l2(x)
		x = self.bn4(x)
		x = self.relu(x)

		x = self.l3(x)
		
		return x
		