import torch.nn as nn
import torchvision.transforms as transforms
import math

from tools import conv2d_Q_fn, linear_Q_fn, activation_quantize_fn

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
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, wbits=32, abits=32):
        super(BasicBlock, self).__init__()
        self.expansion = 1
        Conv2d = conv2d_Q_fn(w_bit=wbits)
        self.act_q = activation_quantize_fn(a_bit=abits)
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(inplanes)
        # relu
        # activation_q
        self.conv1 = Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # relu
        # activation_q
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x.clone()

        x = self.bn1(x)
        x = self.relu(x)
        x = self.act_q(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.act_q(x)
        x = self.conv2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, wbits=32, abits=32):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        Conv2d = conv2d_Q_fn(w_bit=wbits)
        self.act_q = activation_quantize_fn(a_bit=abits)
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(inplanes)
        # relu
        # activation_q
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # activation_q
        self.bn2 = nn.BatchNorm2d(planes)
        # relu
        # activation_q
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        # relu
        # activation_q
        self.conv3 = Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x.clone()

        x = self.bn1(x)
        x = self.relu(x)
        x = self.act_q(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.act_q(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = self.relu(x)
        x = self.act_q(x)
        x = self.conv3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual

        return x


class ResNet_cifar(nn.Module):
    #Default network model: ResNet-20
    def __init__(self, block=BasicBlock, layers=20, wbits=32, abits=32):
        super(ResNet_cifar, self).__init__()
        self.inplanes = 16
        self.wbits = wbits
        self.abits = abits
        n = int((layers - 2) / 6)
        l=n*6+2
        text = f'Creating ResNet-{l}'
        print(text)
        self.Conv2d = conv2d_Q_fn(w_bit=wbits)
        self.Linear = linear_Q_fn(w_bit=wbits)
        self.act_q = activation_quantize_fn(a_bit=abits)

        #first layer not quantized
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        #last layer not quantized
        self.fc = nn.Linear(64, 10, bias=False)
        

        init_model(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.act_q,
                self.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, wbits=self.wbits, abits=self.abits))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, wbits=self.wbits, abits=self.abits))

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
    def __init__(self, layers=18, wbits=32, abits=32):
        super(ResNet_imagenet, self).__init__()
        self.wbits = wbits
        self.abits = abits
        self.Conv2d = conv2d_Q_fn(w_bit=wbits)
        self.act_q = activation_quantize_fn(a_bit=abits)
        self.inplanes = 64

        #first layer not quantized
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
        #last layer not quantized
        self.fc = nn.Linear(512 * block.expansion, 1000, bias=False)

        init_model(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.act_q,
                self.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, wbits=self.wbits, abits=self.abits))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, wbits=self.wbits, abits=self.abits))

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


