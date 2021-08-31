class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(MyModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, 
                      kernel_size=11, stride=4, padding=2),
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




import torchvision.models as models
import torch.optim as optim
vgg = models.vgg16(pretrained=True).to(device)
vgg.classifier[6].out_features = 18
class MyNewNet(nn.Module):
    def __init__(self):
        super(MyNewNet, self).__init__()
        self.vgg = models.vgg16(pretrained = True)
    
    def forward(self, x):
        x = self.vgg(x)
        return x
model = MyNewNet().to(device)
for param in model.features.parameters(): param.requires_grad = False

dtype = torch.floate
ltype = torch.long


model = densenet121(pretrained+True).features.to(device)
for p in model.parameters():
    p.requires_grad = False