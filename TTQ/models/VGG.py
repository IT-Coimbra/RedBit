import torch.nn as nn
from collections import OrderedDict

cfg = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
	#Default: CIFAR-10 (num_classes=10). For ImageNet use num_classes=1000

	def __init__(self, vgg_name='VGG16', num_classes=10, batch_norm=True):
		super(VGG, self).__init__()
		self.num_classes = num_classes

		self.features = self._make_layers(cfg[vgg_name], batch_norm)
		if self.num_classes == 1000:		# ImageNet
			self.classifier = self.classifier = nn.Sequential(OrderedDict([
				('linear1', nn.Linear(25088, 4096, bias=False)),	# --
				('relu1', nn.ReLU(True)),
				('dp1', nn.Dropout()),
				('linear2', nn.Linear(4096, 4096, bias=False)),
				('relu2', nn.ReLU(True)),
				('dp2', nn.Dropout()),
				('linear3', nn.Linear(4096, num_classes, bias=False))
			]))
		elif self.num_classes == 10:		# CIFAR-10
			self.classifier = self.classifier = nn.Sequential(OrderedDict([
				('linear1', nn.Linear(512, 4096, bias=False)),
				('relu1', nn.ReLU(True)),
				('dp1', nn.Dropout()),
				('linear2', nn.Linear(4096, 4096, bias=False)),
				('relu2', nn.ReLU(True)),
				('dp2', nn.Dropout()),
				('linear3', nn.Linear(4096, num_classes, bias=False))
			]))
		else:
			print("Invalid number of classes! Choose 10 for CIFAR-10 or 1000 for ImageNet")
			exit()

	def forward(self, x):
		out = self.features(x)
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		return out

	def _make_layers(self, cfg, batch_norm=False):
		layers = nn.Sequential(OrderedDict())
		in_channels = 3
		i=0
		c=1
		for x in cfg:
			if x == 'M':
				layers.add_module(f'mp{i}', nn.MaxPool2d(kernel_size=2, stride=2))
				i+=1
			else:
				conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False)
				if batch_norm:
					layers.add_module(f'conv{c}', conv2d)
					layers.add_module(f'bn{c}', nn.BatchNorm2d(x))
					layers.add_module(f'relu{c}', nn.ReLU(inplace=True))
				else:
					layers.add_module(f'conv{c}', conv2d)
					layers.add_module(f'relu{c}', nn.ReLU(inplace=True))
				in_channels = x
				c+=1
		if self.num_classes == 1000:		# ImageNet
			layers.add_module('ap1', nn.AdaptiveAvgPool2d((7,7)))
		elif self.num_classes == 10:		# CIFAR-10
			layers.add_module('ap1', nn.AvgPool2d(kernel_size=1, stride=1))
		else:
			print("Invalid number of classes! Choose 10 for CIFAR-10 or 1000 for ImageNet")
			exit()
		return layers