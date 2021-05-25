import torch.nn as nn
from tools import BinActive

cfg = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class Layer(nn.Module):
	def __init__(self, BatchNorm, Conv2d, batch_norm=True, insert_relu=True, binarize_activations=True, fl=False):
		super(Layer, self).__init__()
		self.batch_norm = batch_norm
		self.insert_relu = insert_relu
		self.binarize_activations = binarize_activations
		self.fl = fl

		if self.batch_norm:
			self.bn1 = BatchNorm
		else:
			self.bn1 = lambda x: x

		if self.binarize_activations:
			self.binactiv = BinActive.apply
		else:
			self.binactiv = lambda x: x
		self.conv1 = Conv2d
		
		if self.fl or self.insert_relu:
			self.relu = nn.ReLU(inplace=True)
		else:
			self.relu = lambda x: x

	def forward(self, x):
		if self.fl:
			x = self.conv1(x)
			x = self.bn1(x)
			x = self.relu(x)
		else:
			x = self.bn1(x)
			x = self.binactiv(x)
			x = self.conv1(x)
			x = self.relu(x)

		return x

class Pool(nn.Module):
	def __init__(self, pool_function):
		super(Pool, self).__init__()
		self.pool = pool_function

	def forward(self, x):
		x = self.pool(x)
		return x

class VGG(nn.Module):
	#Default: CIFAR-10 (num_classes=10). For ImageNet use num_classes=1000

	def __init__(self, vgg_name, num_classes=10, batch_norm=True, insert_relu=True, binarize_activations=True):
		super(VGG, self).__init__()
		self.num_classes = num_classes
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

		self.features = self._make_layers(cfg[vgg_name], batch_norm)

		### Classifier ###

		# binactiv
		if self.num_classes == 1000:		# ImageNet
			self.linear1 = nn.Linear(25088, 4096, bias=False)
		elif self.num_classes == 10:		# CIFAR-10
			self.linear1 = nn.Linear(512, 4096, bias=False)
		else:
			print("Invalid number of classes! Choose 10 for CIFAR-10 or 1000 for ImageNet")
			exit()
		# relu
		# binactiv
		self.linear2 = nn.Linear(4096, 4096, bias=False)
		# relu
		self.linear3 = nn.Linear(4096, num_classes, bias=False)


	def forward(self, x):
		x = self.features(x)
		
		x = x.view(x.size(0), -1)
		
		x = self.binactiv(x)
		x = self.linear1(x)
		x = self.relu(x)

		x = self.binactiv(x)
		x = self.linear2(x)
		x = self.relu(x)

		x = self.linear3(x)

		return x

	def _make_layers(self, cfg, batch_norm=False):
		layers = []
		in_channels = 3
		for i,x in enumerate(cfg):
			if i==0:
				conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False)
				batchnorm = nn.BatchNorm2d(x)
				layers.append(Layer(batchnorm, conv2d, batch_norm, self.insert_relu, self.binarize_activations, True))
				in_channels = x
			else:
				if x == 'M':
					layers.append(Pool(nn.MaxPool2d(kernel_size=2, stride=2)))
				else:
					batchnorm = nn.BatchNorm2d(in_channels)
					conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False)
					layers.append(Layer(batchnorm, conv2d, batch_norm, self.insert_relu, self.binarize_activations, False))
					in_channels = x

		if self.num_classes == 1000:		# ImageNet
			layers.append(Pool(nn.AdaptiveAvgPool2d((7,7))))
		elif self.num_classes == 10:		# CIFAR-10
			layers.append(Pool(nn.AvgPool2d(kernel_size=1, stride=1)))
		else:
			print("Invalid number of classes! Choose 10 for CIFAR-10 or 1000 for ImageNet")
			exit()
		return nn.Sequential(*layers)