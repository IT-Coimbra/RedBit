import torch.nn as nn
import torchvision.transforms as transforms
import math
from tools import BinActive

def init_model(model):
	for m in model.modules():
		if isinstance(m, nn.Conv2d):
			n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			m.weight.data.normal_(0, math.sqrt(2. / n))
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1)
			m.bias.data.zero_()


class BasicBlock(nn.Module):
	expansion = 1
	
	def __init__(self, inplanes, planes, stride=1, downsample=None, insert_relu=True, binarize_activations=True):
		super(BasicBlock, self).__init__()
		self.expansion = 1

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

		self.bn1 = nn.BatchNorm2d(inplanes)
		# binactiv
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		# if insert_relu -> ReLU

		self.bn2 = nn.BatchNorm2d(planes)
		#binactiv
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		# if insert_relu -> ReLU

		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x.clone()

		x = self.bn1(x)
		x = self.binactiv(x)
		x = self.conv1(x)
		x = self.relu(x)

		x = self.bn2(x)
		x = self.binactiv(x)
		x = self.conv2(x)

		if self.downsample is not None:
			residual = self.downsample(residual)

		x += residual

		x = self.relu(x)

		return x


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, insert_relu=True, binarize_activations=True):
		super(Bottleneck, self).__init__()
		self.expansion = 4

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

		self.bn1 = nn.BatchNorm2d(inplanes)
		# binactiv
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		# if insert_relu -> ReLU
		
		self.bn2 = nn.BatchNorm2d(planes)
		# binactiv
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		# if insert_relu -> ReLU

		self.bn3 = nn.BatchNorm2d(planes)
		# binactiv
		self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
		# if insert_relu -> ReLU
		
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x.clone()

		x = self.bn1(x)
		x = self.binactiv(x)
		x = self.conv1(x)
		x = self.relu(x)

		x = self.bn2(x)
		x = self.binactiv(x)
		x = self.conv2(x)
		x = self.relu(x)

		x = self.bn3(x)
		x = self.binactiv(x)
		x = self.conv3(x)

		if self.downsample is not None:
			residual = self.downsample(residual)

		x += residual

		x = self.relu(x)

		return x

class Downsample(nn.Module):
	def __init__(self, BatchNorm, Conv2d, binarize_activations=True):
		super(Downsample, self).__init__()
		self.binarize_activations = binarize_activations
		if self.binarize_activations:
			self.binactiv = BinActive.apply
		else:
			self.binactiv = lambda x: x

		self.bn1 = BatchNorm
		#binactiv
		self.conv1 = Conv2d

	def forward(self, x):
		x = self.bn1(x)
		x = self.binactiv(x)
		x = self.conv1(x)

		return x

class ResNet_cifar(nn.Module):
	#Default network model: ResNet-20
	def __init__(self, block=BasicBlock, layers=20, insert_relu=True, binarize_activations=True):
		super(ResNet_cifar, self).__init__()
		self.inplanes = 16
		self.binarize_activations = binarize_activations
		if self.binarize_activations:
			self.binactiv = BinActive.apply
		else:
			self.binactiv = lambda x: x
		self.insert_relu = insert_relu

		n = int((layers - 2) / 6)
		l=n*6+2
		text = f'Creating ResNet-{l}'
		print(text)

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.relu = nn.ReLU(inplace=True)

		self.layer1 = self._make_layer(block, 16, n)
		self.layer2 = self._make_layer(block, 32, n, stride=2)
		self.layer3 = self._make_layer(block, 64, n, stride=2)
		self.avgpool = nn.AvgPool2d(8)
		self.fc = nn.Linear(64, 10, bias=False)

		init_model(self)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			Conv2d = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
			BatchNorm = nn.BatchNorm2d(self.inplanes)
			downsample = Downsample(BatchNorm, Conv2d, self.binarize_activations)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, self.insert_relu, self.binarize_activations))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, insert_relu=self.insert_relu, binarize_activations=self.binarize_activations))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


class ResNet_imagenet(nn.Module):
	#Default network model: ResNet-18
	def __init__(self, layers=18, insert_relu=True, binarize_activations=True):
		super(ResNet_imagenet, self).__init__()
		self.inplanes = 64
		self.binarize_activations = binarize_activations
		if self.binarize_activations:
			self.binactiv = BinActive.apply
		else:
			self.binactiv = lambda x: x
		self.insert_relu = insert_relu

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		if layers==18:
			layers_config=[2,2,2,2]
			block = BasicBlock
		elif layers == 34:
			layers_config=[3,4,6,3]
			block = BasicBlock
		elif layers == 50:
			layers_config=[3,4,6,3]
			block = Bottleneck
		else:
			print('Invalid network depth')
			exit()

		self.layer1 = self._make_layer(block, 64, layers_config[0])
		self.layer2 = self._make_layer(block, 128, layers_config[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers_config[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers_config[3], stride=2)
		self.avgpool = nn.AvgPool2d(7)
		self.fc = nn.Linear(512 * block.expansion, 1000, bias=False)

		init_model(self)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			Conv2d = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
			BatchNorm = nn.BatchNorm2d(self.inplanes)
			downsample = Downsample(BatchNorm, Conv2d, self.binarize_activations)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, self.insert_relu, self.binarize_activations))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, insert_relu=self.insert_relu, binarize_activations=self.binarize_activations))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


