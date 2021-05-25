import torch.nn as nn

# Model for the MNIST dataset
class LeNet_5(nn.Module):
	def __init__(self):
		super(LeNet_5, self).__init__()
		self.act = nn.ReLU(inplace=True)
		
		self.conv1=nn.Conv2d(1, 20, kernel_size=5, stride=1, bias=False)
		self.bn1=nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=False)
		# ReLU
		self.mp1=nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv2=nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0, bias=False)
		self.bn2=nn.BatchNorm2d(50, eps=1e-4, momentum=0.1, affine=True)
		# ReLU
		self.mp2=nn.MaxPool2d(kernel_size=2, stride=2)

		# reshape

		self.linear1=nn.Linear(50*4*4, 500, bias=False)
		self.bn3=nn.BatchNorm1d(500, eps=1e-4, momentum=0.1, affine=True)
		# ReLU

		self.linear2=nn.Linear(500, 250, bias=False)
		self.bn4=nn.BatchNorm1d(250, eps=1e-4, momentum=0.1, affine=True)
		# ReLU

		self.linear3=nn.Linear(250, 10, bias=False)

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

		x = self.linear1(x)
		x = self.bn3(x)
		x = self.act(x)
		
		x = self.linear2(x)
		x = self.bn4(x)
		x = self.act(x)
		
		x = self.linear3(x)
		
		return x
		