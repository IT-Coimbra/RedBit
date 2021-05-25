import torch.nn as nn

cfg = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
	#Default: CIFAR-10 (num_classes=10). For ImageNet use num_classes=1000

	def __init__(self, vgg_name, num_classes=10, batch_norm=True):
		super(VGG, self).__init__()
		self.num_classes = num_classes

		self.features = self._make_layers(cfg[vgg_name], batch_norm)
		if self.num_classes == 10:
			self.classifier = self.classifier = nn.Sequential(
				nn.Linear(512, 4096, bias=False),
				nn.ReLU(True),
				nn.Linear(4096, 4096, bias=False),
				nn.ReLU(True),
				nn.Linear(4096, num_classes, bias=False),
			)
		elif self.num_classes == 1000:
			self.classifier = self.classifier = nn.Sequential(
                nn.Linear(25088, 4096, bias=False),
                nn.ReLU(True),
                nn.Linear(4096, 4096, bias=False),
                nn.ReLU(True),
                nn.Linear(4096, num_classes, bias=False),
            )
		else:
			print("Invalid number of classes! Choose 10 for CIFAR-10 or 1000 for ImageNet")
			exit()

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	def _make_layers(self, cfg, batch_norm=True):
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False)
				if batch_norm:
					layers += [conv2d, nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = x
				
		if self.num_classes == 10:
			layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		elif self.num_classes == 1000:
			layers += [nn.AdaptiveAvgPool2d((7,7))]
		else:
			print("Invalid number of classes! Choose 10 for CIFAR-10 or 1000 for ImageNet")
			exit()
		return nn.Sequential(*layers)