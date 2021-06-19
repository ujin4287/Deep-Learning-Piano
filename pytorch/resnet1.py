import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResidualBlock(nn.Module):
    expansion = 4  ### 왜 4로 해줬는지 잘 이해가 안갑니다..

    def __init__(self, inplanes, planes, stride=1, down_sample=None):  # inplanes=in_channel / planes=out_channel
        super(ResidualBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)

        out += shortcut
        out = self.relu(out)

        return out


class ResNet(nn.Module):  # ResNet(block,[3,4,6,3],10,True)
    def __init__(self, block, num_layers, num_classes=36, zero_init_residual=True): # ResNet(block, [3, 4, 6, 3], 10, True)
        super(ResNet, self).__init__()
        self.inplanes = 480

        # batch,3x32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # batch,32x32x32

        self.bn1 = nn.BatchNorm2d(32)  # out_channels
        self.relu = nn.ReLU(inplace=True)

        # 32x480x480
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 32x240x240

        self.layer1 = self._make_layer(block, 32, num_layers[0], stride=1)
        # 128x240x240
        self.layer2 = self._make_layer(block, 64, num_layers[1], stride=1)
        # 256x240x240
        self.layer3 = self._make_layer(block, 128, num_layers[2], stride=2)
        # 512x120x120
        self.layer4 = self._make_layer(block, 256, num_layers[3], stride=2)
        # 1024x60x60

        ### 그 ResNet에서 일반적으로 연산량을 맞춰주기 위해서 feature map 의 크기가 절반으로 줄때 개수를 2배로

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        # 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResidualBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # 1 self.inplanes = 32
    # 1 self.layer1 = self._make_layer(Bottleneck, 32, 3, 1)

    # 2 self.inplanes = 128
    # 2 self.layer1 = self._make_layer(Bottleneck, 64, 4, 2)

    def _make_layer(self, block, planes, blocks, stride=1):  ### planes 가 깊이 말하는 건가요?
        down_sample = None

        if stride != 1 or self.inplanes != planes * block.expansion:  # 64 != 64 * 4 =256   ### down_sampling하는 부분 한번만 설명 가능할까요?
            # down sample은 stride2 일때 서로 달라지는 스킵패스로 가는 identity하고 아웃하고 사아즈가 달라질때 맞춰주기 위해 사용한다.
            # ResNet에서는 특이하게 downsample을 이용해서 채널을 맞춰주기 위해서 사용을한다.
            down_sample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, down_sample))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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


def resnet():
    block = ResidualBlock
    model = ResNet(block, [3, 4, 6, 3], 36, True) # (self, block, num_layers, num_classes=36, zero_init_residual=True)
    return model