import torch.nn as nn
from tools import QLinear, QConv2d

class LeNet_5(nn.Module):
	def __init__(self, wbits=32, abits=32):
		super(LeNet_5, self).__init__()
		self.abits=abits
		self.wbits=wbits
		if self.abits==32:
			self.act = nn.ReLU(inplace=True)
		else:
			self.act=nn.Hardtanh(inplace=True)

		self.conv1=QConv2d(abits=self.abits, wbits=self.wbits, in_channels=1, out_channels=20, kernel_size=5, stride=1, bias=False)
		self.bn1=nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=False)
		# activation
		self.mp1=nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv2=QConv2d(abits=self.abits, wbits=self.wbits, in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0, bias=False)
		self.bn2=nn.BatchNorm2d(50, eps=1e-4, momentum=0.1, affine=True)
		# activation
		self.mp2=nn.MaxPool2d(kernel_size=2, stride=2)

		# reshape

		self.l1=QLinear(abits=self.abits, wbits=self.wbits, in_features=50*4*4, out_features=500, bias=False)
		self.bn3=nn.BatchNorm1d(500, eps=1e-4, momentum=0.1, affine=True)
		# activation

		self.l2=QLinear(abits=self.abits, wbits=self.wbits, in_features=500, out_features=250, bias=False)
		self.bn4=nn.BatchNorm1d(250, eps=1e-4, momentum=0.1, affine=True)
		# activation

		self.l3=QLinear(abits=self.abits, wbits=self.wbits, in_features=250, out_features=10, bias=False)
	

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.act(x)
		x = self.mp1(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.act(x)
		x = self.mp2(x)

		x = x.view(x.size(0), -1)

		x = self.l1(x)
		x = self.bn3(x)
		x = self.act(x) 
		
		x = self.l2(x)
		x = self.bn4(x)
		x = self.act(x)
		
		x= x = self.l3(x)

		return x
		