import torch
import torch.nn as nn
import numpy as np
import timm

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(MyModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def init_param(self): # 파라미터 초기화
        for m in self.modules():
            if isinstance(m,nn.Conv2d): # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d): # init BN
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)


class Special_Mission(nn.Module): # 모델 이해하기, 반복이 많음 -> 파이써닉하게 다시 바꾸어야함
    def __init__(self, num_classes):
        super(Special_Mission, self).__init__()

        self.num_classes = num_classes
        self.model_zip = list()

        self.conv1_1 = self.conv_batch(3, 32)
        self.conv1_2 = self.conv_batch(32, 64, stride=2)
        self.res1 = self.res_layer_iter(64, 1)
        self.conv2_1 = self.conv_batch(64, 128, stride=2)
        self.res2 = self.res_layer_iter(128, 2) # foward_residual * 2
        self.conv3_1 = self.conv_batch(128, 256, stride=2)
        self.res3 = self.res_layer_iter(256, 8) # foward_residual * 8
        self.conv4_1 = self.conv_batch(256, 512, stride=2)
        self.res4 = self.res_layer_iter(512, 8) # foward_residual * 8
        self.conv5_1 = self.conv_batch(512, 1024, stride=2)
        self.res5 = self.res_layer_iter(1024, 4) # foward_residual * 4
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)
        
        # 가중치 불러오기용 # 정리해야함
        self.model_zip.append([self.conv1_1[0], self.conv1_1[1]])
        self.model_zip.append([self.conv1_2[0], self.conv1_2[1]])
        for j in range(2):
            self.model_zip.append([self.res1[0][j][0], self.res1[0][j][1]])
        self.model_zip.append([self.conv2_1[0], self.conv2_1[1]])
        for i in range(2):
            for j in range(2):
                self.model_zip.append([self.res2[i][j][0], self.res2[i][j][1]])
        self.model_zip.append([self.conv3_1[0], self.conv3_1[1]])
        for i in range(8):
            for j in range(2):
                self.model_zip.append([self.res3[i][j][0], self.res3[i][j][1]])
        self.model_zip.append([self.conv4_1[0], self.conv4_1[1]])
        for i in range(8):
            for j in range(2):
                self.model_zip.append([self.res4[i][j][0], self.res4[i][j][1]])
        self.model_zip.append([self.conv5_1[0], self.conv5_1[1]])
        for i in range(4):
            for j in range(2):
                self.model_zip.append([self.res5[i][j][0], self.res5[i][j][1]])

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        
        # res_1
        residual = x
        x = self.res1[0][0](x)
        x = self.res1[0][1](x)
        x += residual
        
        x = self.conv2_1(x)
        
        # res_2
        for i in range(2):
            residual = x
            x = self.res2[i][0](x)
            x = self.res2[i][1](x)
            x += residual
        
        x = self.conv3_1(x)
        
        # res_3
        for i in range(8):
            residual = x
            x = self.res3[i][0](x)
            x = self.res3[i][1](x)
            x += residual
        
        x = self.conv4_1(x)
        
        # res_4
        for i in range(8):
            residual = x
            x = self.res4[i][0](x)
            x = self.res4[i][1](x)
            x += residual
        
        x = self.conv5_1(x)
        
        # res_5
        for i in range(4):
            residual = x
            x = self.res5[i][0](x)
            x = self.res5[i][1](x)
            x += residual
        
        x = self.global_avg_pool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)

        return x
    
    def conv_batch(self, in_num, out_num, kernel_size=3, padding=1, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), # 실제 구현에서 bias를 안씀 yolo53
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU())
    
    def res_layer_iter(self, in_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            res1 = self.conv_batch(in_channels, in_channels//2, kernel_size=1, padding=0) # resnet 안의 conv_batch
            res2 = self.conv_batch(in_channels//2, in_channels)
            layers.append(nn.ModuleList([res1, res2]))
        return nn.ModuleList(layers)
    
    def init_param(self): # 파라미터 초기화
        for m in self.modules():
            if isinstance(m,nn.Conv2d): # init conv
                nn.init.kaiming_normal_(m.weight)
#                 nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d): # init BN
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def weight_allocate(self, conv_layer, bn_layer, ptr, weights):
        # batchnorm weight
        num_b = bn_layer.bias.numel()  # Number of biases
        # Bias
        bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
        bn_layer.bias.data.copy_(bn_b)
        ptr += num_b
        # Weight
        bn_w = torch.from_numpy(
            weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
        bn_layer.weight.data.copy_(bn_w)
        ptr += num_b
        # Running Mean
        bn_rm = torch.from_numpy(
            weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
        bn_layer.running_mean.data.copy_(bn_rm)
        ptr += num_b
        # Running Var
        bn_rv = torch.from_numpy(
            weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
        bn_layer.running_var.data.copy_(bn_rv)
        ptr += num_b

        # conv weight
        num_w = conv_layer.weight.numel()
        conv_w = torch.from_numpy(
            weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
        conv_layer.weight.data.copy_(conv_w)
        ptr += num_w
        return conv_layer, bn_layer, ptr
            
    def load_weights(self, weight_path): # darknet53.conv.74 파일
        # Open the weights file
        with open(weight_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            header_info = header  # Needed to write header when saving weights
            seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights
            
        ptr = 0
        for conv_layer, bn_layer in self.model_zip: # conv74
            conv_layer, bn_layer, ptr = self.weight_allocate(conv_layer, bn_layer, ptr, weights)
            
            
class nfnet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(nfnet, self).__init__()
        self.net = timm.create_model("nfnet_f7s", pretrained=True, num_classes= num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    
    # timm_models = timm.list_models("*") timm 지원 모델 목록
    
#     def init_param(self): # 파라미터 초기화할 필요가 없음 pretrained 쓸거임
#         for m in self.modules():
#             if isinstance(m,nn.Conv2d): # init conv
#                 nn.init.kaiming_normal_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m,nn.BatchNorm2d): # init BN
#                 nn.init.constant_(m.weight,1)
#                 nn.init.constant_(m.bias,0)
#             elif isinstance(m,nn.Linear): # lnit dense
#                 nn.init.kaiming_normal_(m.weight)
#                 nn.init.zeros_(m.bias)

class efficient_b0(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(efficient_b0, self).__init__()
        self.net = timm.create_model("efficientnet_b0", pretrained=True, num_classes= num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    
    # timm_models = timm.list_models("*") timm 지원 모델 목록
    
#     def init_param(self): # 파라미터 초기화할 필요가 없음 pretrained 쓸거임
#         for m in self.modules():
#             if isinstance(m,nn.Conv2d): # init conv
#                 nn.init.kaiming_normal_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m,nn.BatchNorm2d): # init BN
#                 nn.init.constant_(m.weight,1)
#                 nn.init.constant_(m.bias,0)
#             elif isinstance(m,nn.Linear): # lnit dense
#                 nn.init.kaiming_normal_(m.weight)
#                 nn.init.zeros_(m.bias)

class efficient_b4(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(efficient_b4, self).__init__()
        self.net = timm.create_model("efficientnet_b4", pretrained=True, num_classes= num_classes, drop_rate=0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    
    # timm_models = timm.list_models("*") timm 지원 모델 목록
    
#     def init_param(self): # 파라미터 초기화할 필요가 없음 pretrained 쓸거임
#         for m in self.modules():
#             if isinstance(m,nn.Conv2d): # init conv
#                 nn.init.kaiming_normal_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m,nn.BatchNorm2d): # init BN
#                 nn.init.constant_(m.weight,1)
#                 nn.init.constant_(m.bias,0)
#             elif isinstance(m,nn.Linear): # lnit dense
#                 nn.init.kaiming_normal_(m.weight)
#                 nn.init.zeros_(m.bias)