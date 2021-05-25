import torch.nn as nn
import torchvision.transforms as transforms
import math

from tools import QConv2d, QLinear


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
	
	def __init__(self, abits, wbits, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.expansion = 1
		self.abits=abits
		self.wbits=wbits
		if self.abits==32:
			self.act = nn.ReLU(inplace=True)
		else:
			self.act=nn.Hardtanh(inplace=True)
		
		self.conv1 = QConv2d(abits=self.abits, wbits=self.wbits, in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)

		self.conv2 = QConv2d(abits=self.abits, wbits=self.wbits, in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x.clone()

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.act(x)

		x = self.conv2(x)
		x = self.bn2(x)

		if self.downsample is not None:
			residual = self.downsample(residual)

		x += residual
		x = self.act(x)

		return x


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, abits, wbits, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.expansion = 4
		self.abits=abits
		self.wbits=wbits
		if self.abits==32:
			self.act = nn.ReLU(inplace=True)
		else:
			self.act=nn.Hardtanh(inplace=True)

		self.conv1 = QConv2d(abits=self.abits, wbits=self.wbits, in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		
		self.conv2 = QConv2d(abits=self.abits, wbits=self.wbits, in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		
		self.conv3 = QConv2d(abits=self.abits, wbits=self.wbits, in_channels=planes, out_channels=planes * self.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x.clone()

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.act(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.act(x)

		x = self.conv3(x)
		x = self.bn3(x)

		if self.downsample is not None:
			residual = self.downsample(residual)

		x += residual
		x = self.act(x)

		return x


class ResNet_cifar(nn.Module):
	#Default network model: ResNet-20
	def __init__(self, abits=32, wbits=32, block=BasicBlock, layers=20):
		super(ResNet_cifar, self).__init__()
		self.inplanes = 16
		n = int((layers - 2) / 6)
		l=n*6+2
		text = f'Creating ResNet-{l}'
		print(text)

		self.abits=abits
		self.wbits=wbits
		if self.abits==32:
			self.act = nn.ReLU(inplace=True)
		else:
			self.act=nn.Hardtanh(inplace=True)
		
		self.conv1 = QConv2d(abits=self.abits, wbits=self.wbits, in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)

		self.layer1 = self._make_layer(block, 16, n)
		self.layer2 = self._make_layer(block, 32, n, stride=2)
		self.layer3 = self._make_layer(block, 64, n, stride=2)
		self.avgpool = nn.AvgPool2d(8)
		
		self.fc = QLinear(abits=self.abits, wbits=self.wbits, in_features=64, out_features=10, bias=False)

		init_model(self)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				QConv2d(abits=self.abits, wbits=self.wbits, in_channels=self.inplanes, out_channels=planes * block.expansion, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.abits, self.wbits, self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.abits, self.wbits, self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)

		x = self.act(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)

		x = self.act(x)
		x = self.fc(x)

		return x


class ResNet_imagenet(nn.Module):
	#Default network model: ResNet-18
	def __init__(self, abits=32, wbits=32, layers=18):
		super(ResNet_imagenet, self).__init__()
		self.inplanes = 64

		self.abits=abits
		self.wbits=wbits
		if self.abits==32:
			self.act = nn.ReLU(inplace=True)
		else:
			self.act=nn.Hardtanh(inplace=True)

		self.conv1 = QConv2d(abits=self.abits, wbits=self.wbits, in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
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
		self.fc = QLinear(abits=self.abits, wbits=self.wbits, in_features=512 * block.expansion, out_features=1000, bias=False)

		init_model(self)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				QConv2d(abits=self.abits, wbits=self.wbits, in_channels=self.inplanes, out_channels=planes * block.expansion, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.abits, self.wbits, self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.abits, self.wbits, self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.act(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.act(x)
		x = self.fc(x)

		return x


