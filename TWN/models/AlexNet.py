import torch.nn as nn

class AlexNet(nn.Module):
	def __init__(self):
		super(AlexNet, self).__init__()
		self.act = nn.ReLU(inplace=True)

		self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0, bias=False)
		self.bn1 = nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True)
		self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)
		# ReLU

		self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=1, bias=False)
		self.bn2 = nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=True)
		self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2)
		# ReLU

		self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(384, eps=1e-4, momentum=0.1, affine=True)
		# ReLU

		self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
		self.bn4 = nn.BatchNorm2d(384, eps=1e-4, momentum=0.1, affine=True)
		# ReLU

		self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
		self.bn5 = nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=True)
		self.mp3 = nn.MaxPool2d(kernel_size=3, stride=2)
		# ReLU

		#reshape

		self.lin1 = nn.Linear(256*6*6, 4096, bias=False)
		self.bn6 = nn.BatchNorm1d(4096, eps=1e-4, momentum=0.1, affine=True)
		# ReLU

		self.dp1 = nn.Dropout(0.5)
		self.lin2 = nn.Linear(4096, 4096, bias=False)
		self.bn7 = nn.BatchNorm1d(4096, eps=1e-4, momentum=0.1, affine=True)
		# ReLU

		self.dp2 = nn.Dropout(0.5)
		self.lin3 = nn.Linear(4096, 1000, bias=False)
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

		x = self.dp1(x)
		x = self.lin2(x)
		x = self.bn7(x)
		x = self.act(x)

		x = self.dp2(x)
		x = self.lin3(x)
		x = self.bn8(x)

		return x