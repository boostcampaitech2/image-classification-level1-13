import os
import pandas as pd
import dataset
import torch
import numpy as np
import model
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from PIL import Image

# meta 데이터와 이미지 경로를 불러옵니다.
test_dir = "../../input/data/eval"
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'images')

# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
transform = transforms.Compose([
    transforms.CenterCrop((400,300)),
    Resize((224, 224)),
    ToTensor()
])

dataset_ = dataset.Dataset(image_paths, transform, train=False)

loader = DataLoader(
    dataset_,
    shuffle=False
)

# 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
Weight_PATH = "./save_weights/model_weight5.pt"
device = torch.device('cuda')
model_ = model.nfnet(num_classes=18).to(device)
model_.load_state_dict(torch.load(Weight_PATH))
model_.eval()

# 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
all_predictions = []
for images in loader:
    with torch.no_grad():
        images = images.to(device)
        pred = model_(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
print('test inference is done!')