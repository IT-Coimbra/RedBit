import torch.nn as nn
from tools import BinActive

class AlexNet(nn.Module):
	def __init__(self, insert_relu=True, binarize_activations=True):
		super(AlexNet, self).__init__()
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

		self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0, bias=False)
		self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)
		self.act1 = nn.ReLU(inplace=True)
		self.bn1 = nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True)

		self.bn2 = nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True)
		# binactiv
		self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=1, bias=False)
		# if insert_relu -> ReLU
		self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2)

		self.bn3 = nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=True)
		# binactiv
		self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=False)
		# if insert_relu -> ReLU

		self.bn4 = nn.BatchNorm2d(384, eps=1e-4, momentum=0.1, affine=True)
		# binactiv
		self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
		# if insert_relu -> ReLU

		self.bn5 = nn.BatchNorm2d(384, eps=1e-4, momentum=0.1, affine=True)
		# binactiv
		self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
		# if insert_relu -> ReLU
		self.mp3 = nn.MaxPool2d(kernel_size=3, stride=2)

		#reshape

		self.bn6 = nn.BatchNorm1d(256*6*6, eps=1e-4, momentum=0.1, affine=True)
		# binactiv
		self.lin1 = nn.Linear(256*6*6, 4096, bias=False)
		# if insert_relu -> ReLU

		self.bn7 = nn.BatchNorm1d(4096, eps=1e-4, momentum=0.1, affine=True)
		# binactiv
		self.lin2 = nn.Linear(4096, 4096, bias=False)
		# if insert_relu -> ReLU

		self.bn8 = nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True)
		self.lin3 = nn.Linear(4096, 1000, bias=False)

	def forward(self, x):
		x = self.conv1(x)
		x = self.mp1(x)
		x = self.bn1(x)
		x = self.act1(x)

		x = self.bn2(x)
		x = self.binactiv(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.mp2(x)

		x = self.bn3(x)
		x = self.binactiv(x)
		x = self.conv3(x)
		x = self.relu(x)

		x = self.bn4(x)
		x = self.binactiv(x)
		x = self.conv4(x)
		x = self.relu(x)

		x = self.bn5(x)
		x = self.binactiv(x)
		x = self.conv5(x)
		x = self.relu(x)
		x = self.mp3(x)

		x = x.view(x.size(0), -1)

		x = self.bn6(x)
		x = self.binactiv(x)
		x = self.lin1(x)
		x = self.relu(x)

		x = self.bn7(x)
		x = self.binactiv(x)
		x = self.lin2(x)
		x = self.relu(x)

		x = self.bn8(x)
		x = self.lin3(x)

		return x