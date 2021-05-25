import torch.nn as nn

from tools import QConv2d, QLinear

cfg = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
	#Default: CIFAR-10 (num_classes=10). For ImageNet use num_classes=1000

	def __init__(self, vgg_name, abits=32, wbits=32, num_classes=10, batch_norm=True):
		super(VGG, self).__init__()
		self.num_classes = num_classes
		self.abits=abits
		self.wbits=wbits
		if self.abits==32:
			self.act = nn.ReLU(inplace=True)
		else:
			self.act=nn.Hardtanh(inplace=True)

		self.features = self._make_layers(cfg[vgg_name], batch_norm)
		if self.num_classes == 1000:		# ImageNet
			self.classifier = self.classifier = nn.Sequential(
				self.act,
				QLinear(abits=self.abits, wbits=self.wbits, in_features=25088, out_features=4096, bias=False), # --
				self.act,
				QLinear(abits=self.abits, wbits=self.wbits, in_features=4096, out_features=4096, bias=False),
				self.act,
				QLinear(abits=self.abits, wbits=self.wbits, in_features=4096, out_features=self.num_classes, bias=False),
			)
		elif self.num_classes == 10:		# CIFAR-10
			self.classifier = self.classifier = nn.Sequential(
				self.act,
				QLinear(abits=self.abits, wbits=self.wbits, in_features=512, out_features=4096, bias=False),
				self.act,
				QLinear(abits=self.abits, wbits=self.wbits, in_features=4096, out_features=4096, bias=False),
				self.act,
				QLinear(abits=self.abits, wbits=self.wbits, in_features=4096, out_features=self.num_classes, bias=False),
			)
		else:
			print("Invalid number of classes! Choose 10 for CIFAR-10 or 1000 for ImageNet")
			exit()

	def forward(self, x):
		out = self.features(x)
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		return out

	def _make_layers(self, cfg, batch_norm=True):
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				conv2d = QConv2d(abits=self.abits, wbits=self.wbits, in_channels=in_channels, out_channels=x, 
									kernel_size=3, padding=1, bias=False)
				if batch_norm:
					layers += [conv2d, nn.BatchNorm2d(x), self.act]
				else:
					layers += [conv2d, self.act]
				in_channels = x
		if self.num_classes == 1000:		# ImageNet
			layers += [nn.AdaptiveAvgPool2d((7,7))]
		elif self.num_classes == 10:		# CIFAR-10
			layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		else:
			print("Invalid number of classes! Choose 10 for CIFAR-10 or 1000 for ImageNet")
			exit()
		return nn.Sequential(*layers)