import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)



##############################################
# Darknet
def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet(nn.Module):
    def __init__(self, block=DarkResidualBlock, num_classes:int = 18):
        super(Darknet, self).__init__()

        self.num_classes = num_classes

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


def darknet(num_classes):
    return Darknet(DarkResidualBlock, num_classes)
##############################################

class MyResnet(nn.Module):
    def __init__(self, num_classes):
        super(MyResnet, self).__init__()
        self.layer = torchvision.models.resnet18(pretrained=True)
        self.layer.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.layer(x)


class MyVGG(nn.Module):
    def __init__(self, num_classes):
        super(MyVGG, self).__init__()
        self.layer = torchvision.models.vgg11_bn(pretrained=True)
        self.layer.classifier[6] = nn.Linear(in_features=4096, out_features=18, bias=True)

    def forward(self, x):
        return self.layer(x)


class MyGoogleNet(nn.Module):
    def __init__(self, num_classes):
        super(MyGoogleNet, self).__init__()
        self.layer = torchvision.models.googlenet(pretrained=True)
        self.layer.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.layer(x)


class MyInceptionNet(nn.Module):
    def __init__(self, num_classes):
        super(MyInceptionNet, self).__init__()
        self.layer = torchvision.models.inception_v3(pretrained=True)
        self.layer.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.layer(x)


class MyWideResnet(nn.Module):
    def __init__(self, num_classes):
        super(MyWideResnet, self).__init__()
        self.layer = torchvision.models.wide_resnet50_2(pretrained=True)
        self.layer.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.layer(x)


class MyClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MyClassifier, self).__init__()
        self.mask_classifier = MyResnet(num_classes=3)
        self.gender_classifier = MyResnet(num_classes=2)
        self.age_classifier = MyResnet(num_classes=3)

    def forward(self, x):
        mask = self.mask_classifier(x)
        gender = self.gender_classifier(x)
        age = self.age_classifier(x)
        
        out = torch.mm(mask, gender)
        
        return out


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요.
        2. 결과로 나온 output 을 return 해주세요.
        """
        return x


if __name__ == "__main__":
    # a = torch.tensor([0, 0]).unsqueeze(-1)
    # b = torch.tensor([1, 1]).unsqueeze(-1)
    # print(a)
    # print(b)
    # print(torch.cat((a,b)))
    # print(torch.cat((a,b), dim=1))
    data = torch.randn(size=(100,3,128,96))
    clsfier = MyGoogleNet(18)
    print(clsfier.layer)

    # res = MyResnet(18)
    # print(res(data))
    # print(res(data).size())


