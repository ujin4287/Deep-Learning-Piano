import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(IdentityPadding, self).__init__()

        self.pooling = nn.MaxPool2d(1, stride=stride)
        self.add_channels = out_channels - in_channels  ### 무슨역할을 하는 부분이죠?

    def forward(self, x):
        ### feature map의 마지막 축에 대해서는 (0, 0)으로 padding하고 마지막에서 두 번째 축에 대해서는 (0, 0), 그리고 마지막에서 세 번째 축은 (0, self.add_channels)로 padding하라는 뜻이다
        ### 따라서 channels 축에 대해서 한 방향으로 self.add_channels만큼 padding이 될 것이다.
        ###### 위에 말이 무슨 뜻인지 모르겠습니다...

        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        out = self.pooling(out)
        return out


class ResidualBlock(nn.Module):
    ### 여기서 stride=1, down_sample=False 라고 하는데 주는 값이 2, True 인데 init이 끝나고 1, False로 바뀐다는 것인가요?
    def __init__(self, in_channels, out_channels, stride=1, down_sample=False): # 첫번째 conv 의 in_channel, out_channel
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        if down_sample: # down sample 시 padding = 0, stride 1?
            self.down_sample = IdentityPadding(in_channels, out_channels, stride)
        else:
            self.down_sample = None

    def forward(self, x): ### x가 뭐죠?
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)

        out += shortcut         ### 무슨 의미죠? element-wise addition 이라는데 왜 해주는 거죠? @@@ skip connection
        out = self.relu(out)
        return out


class ResNet(nn.Module): # torch.Size([3, 3, 500, 500])
    def __init__(self, num_layers, block, num_classes=36):
        super(ResNet, self).__init__()
        self.num_layers = num_layers

        # RGB이므로 channel 수 3개, kernel size 3, padding 1, stride 1이므로 feature map의 사이즈 유지
        # 3*32*32 -> 16*32*32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                               stride=1, padding=1, bias=False)
        # BatchNorm 역할..?
        # weight 값이 가중이 되어서 Hidden Node의 값이 변하는 것이 아니라 변하는 범위가 작으면 학습이 잘 될 것
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # 2n개씩 구분한 이유 2n개의 layer마다 feature map의 사이즈가 반이 되고 channel 수는 2배가 되기 때문
        # Feature-map의 크기가 절반으로 작아지는 경우는 연산량의 균형을 맞추기 위해 필터의 수를 두 배로 늘린다.
        # block = residual block
        # feature map size = 32x32x16
        self.layers_2n = self.get_layers(block, 16, 16, stride=1)   #block, in_channels, out_channels, stride
        # feature map size = 16x16x32
        self.layers_4n = self.get_layers(block, 16, 32, stride=2)
        # feature map size = 8x8x64
        self.layers_6n = self.get_layers(block, 32, 64, stride=2)

        # output layers
        self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.fc_out = nn.Linear(817216, num_classes) # 64?

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # block을 가지고 2n개의 layer를 만드는 get_layer 함수

		###### 왜 이부분이 2n개의 layer를 만드는 부분인가요???

    def get_layers(self, block, in_channels, out_channels, stride):

        # 비교해서 stride 를 2일때 down sample을 한다.(이 경우가 위에서 in, out channel 다를 때)
        if stride == 2:
            down_sample = True
        else:
            down_sample = False

        layers_list = nn.ModuleList(
            [block(in_channels, out_channels, stride, down_sample)])

        for _ in range(self.num_layers - 1):
            layers_list.append(block(out_channels, out_channels))

        return nn.Sequential(*layers_list)

    # 호출?
    def forward(self, x):
        # 밑의 3개는 모든 ResNet에서 동일하다.
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layers_2n(x)
        x = self.layers_4n(x)
        x = self.layers_6n(x)

        # print(x.shape) # torch.Size([3, 64, 120, 120])
        x = self.avg_pool(x)
        # print(x.shape) # torch.Size([3, 64, 113, 113])
        x = x.view(x.size(0), -1)
        # print(x.shape) # torch.Size([3, 817216])
        x = self.fc_out(x)
        x = torch.sigmoid(x)
        return x


def resnet():
    block = ResidualBlock
    # total number of layers if 6n + 2. if n is 5 then the depth of network is 32.
    ### 왜 32인가요? 2n+2n+2n+ pooling + fc 라서 그런건가요?
    model = ResNet(3, block)    # n값
    # n은 3, 5, 7, 9, 18 중에 하나를 사용한다(논문에서 그렇다. 다른 숫자를 사용해도 무방하다). 각각 ResNet-20, ResNet-32, ResNet-44, ResNet-56, ResNet-110에 해당
    return model
