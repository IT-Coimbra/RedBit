import torch.nn as nn
from tools import BinActive

class LeNet_5(nn.Module):
	def __init__(self, insert_relu=True, binarize_activations=True):
		super(LeNet_5, self).__init__()
		self.binarize_activations = binarize_activations
		if self.binarize_activations:
			self.binactiv = BinActive.apply
		else:
			self.binactiv = lambda x: x
		self.insert_relu = insert_relu
		if self.insert_relu:
			self.relu = nn.ReLU(inplace=True)
		else:
			self.relu = lambda x: x
		
		self.conv1=nn.Conv2d(1, 20, kernel_size=5, stride=1, bias=False)
		self.bn1=nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=False)
		self.act1=nn.ReLU(inplace=True)
		self.mp1=nn.MaxPool2d(kernel_size=2, stride=2)

		self.bn2=nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=True)
		# binactiv
		self.conv2=nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0, bias=False)
		# if insert_relu -> ReLU else lambda x: x
		self.mp2=nn.MaxPool2d(kernel_size=2, stride=2)

		self.bn3=nn.BatchNorm1d(50*4*4, eps=1e-4, momentum=0.1, affine=True)
		# binactiv
		self.l1=nn.Linear(50*4*4, 500, bias=False)
		# if insert_relu -> ReLU else lambda x: x

		self.bn4=nn.BatchNorm1d(500, eps=1e-4, momentum=0.1, affine=True)
		# binactiv
		self.l2=nn.Linear(500, 250, bias=False)
		# if insert_relu -> ReLU else lambda x: x

		self.l3=nn.Linear(250, 10, bias=False)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.act1(x)
		x = self.mp1(x)

		x = self.bn2(x)
		x = self.binactiv(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.mp2(x)

		x = x.view(x.size(0), -1)

		x = self.bn3(x)
		x = self.binactiv(x)
		x = self.l1(x)
		x = self.relu(x)
		
		x = self.bn4(x)
		x = self.binactiv(x)
		x = self.l2(x)
		x = self.relu(x)
		
		x = self.l3(x)
		
		return x
		