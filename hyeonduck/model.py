import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm



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



class shufflenet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        
        self.net = models.shufflenet_v2_x1_0(pretrained=True)
        self.net.fc = nn.Linear(1024, num_classes)

    
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
  
        return self.net(x)

class inception_v3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
                
        self.net = models.inception_v3(pretrained = True)
        self.net.AuxLogits.fc = nn.Linear(768, num_classes)
        self.net.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
  
        return self.net(x)

class Squeeznet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
                
        self.net = models.squeezenet1_0(pretrained = True)
        self.net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1),
                                           stride=(1,1))
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
  
        return self.net(x)




class resnext(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(resnext, self).__init__()
        
        self.net = timm.create_model("ig_resnext101_32x16d", pretrained=True, num_classes= num_classes, drop_rate=0.7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x  

class vit16(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(vit16, self).__init__()

        self.net = timm.create_model("vit_small_patch16_224", pretrained = True, num_classes = num_classes, drop_rate = 0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x  