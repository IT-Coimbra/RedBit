import torch.nn as nn
from tools import QLinear, QConv2d

class AlexNet(nn.Module):
	def __init__(self, wbits=32, abits=32):
		super(AlexNet, self).__init__()
		self.abits=abits
		self.wbits=wbits
		if self.abits==32:
			self.act = nn.ReLU(inplace=True)
		else:
			self.act=nn.Hardtanh(inplace=True)

		self.conv1 = QConv2d(abits=self.abits, wbits=self.wbits, in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0, bias=False)
		self.bn1 = nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True)
		self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)
		# activation

		self.conv2 = QConv2d(abits=self.abits, wbits=self.wbits, in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, groups=1, bias=False)
		self.bn2 = nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=True)
		self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2)
		# activation

		self.conv3 = QConv2d(abits=self.abits, wbits=self.wbits, in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(384, eps=1e-4, momentum=0.1, affine=True)
		# activation
		
		self.conv4 = QConv2d(abits=self.abits, wbits=self.wbits, in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
		self.bn4 = nn.BatchNorm2d(384, eps=1e-4, momentum=0.1, affine=True)
		# activation

		self.conv5 = QConv2d(abits=self.abits, wbits=self.wbits, in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
		self.bn5 = nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=True)
		self.mp3 = nn.MaxPool2d(kernel_size=3, stride=2)
		# activation

		#reshape

		self.lin1 = QLinear(abits=self.abits, wbits=self.wbits, in_features=256*6*6, out_features=4096, bias=False)
		self.bn6 = nn.BatchNorm1d(4096, eps=1e-4, momentum=0.1, affine=True)
		# activation

		self.lin2 = QLinear(abits=self.abits, wbits=self.wbits, in_features=4096, out_features=4096, bias=False)
		self.bn7 = nn.BatchNorm1d(4096, eps=1e-4, momentum=0.1, affine=True)
		# activation

		self.lin3 = QLinear(abits=self.abits, wbits=self.wbits, in_features=4096, out_features=1000, bias=False)
		self.bn8 = nn.BatchNorm1d(1000, eps=1e-3, momentum=0.1, affine=True)

	def forward(self, x):
		
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.mp1(x)
		x = self.act(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.mp2(x)
		x = self.act(x)

		x = self.conv3(x)
		x = self.bn3(x)
		x = self.act(x)

		x = self.conv4(x)
		x = self.bn4(x)
		x = self.act(x)

		x = self.conv5(x)
		x = self.bn5(x)
		x = self.mp3(x)
		x = self.act(x)

		x = x.view(x.size(0), -1)

		x = self.lin1(x)
		x = self.bn6(x)
		x = self.act(x)

		x = self.lin2(x)
		x = self.bn7(x)
		x = self.act(x)

		x = self.lin3(x)
		x = self.bn8(x)

		return x