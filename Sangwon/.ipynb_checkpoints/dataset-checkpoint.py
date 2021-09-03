from torch.utils.data import Dataset
from PIL import Image
# 얼굴인식
from facenet_pytorch import MTCNN
import numpy as np
import torch
import cv2

############################# Normal Detect ###################################
class Dataset(Dataset):
    def __init__(self, img_paths, transform, class_=None, train=False):
        self.img_paths = img_paths
        self.transform = transform
        self.class_ = class_
        self.train = train

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        
        if self.transform:
            image = self.transform(image)
        if self.train == True:
            class_label = self.class_[index]
            return image, class_label
        else:
            return image

    def __len__(self):
        return len(self.img_paths)

############################### Face Detect ###################################
class FaceDataset(Dataset):
    def __init__(self, img_paths, transform, class_=None, train=False):
        self.img_paths = img_paths
        self.transform = transform
        self.class_ = class_
        self.train = train

    def __getitem__(self, index):
        image = self.img_paths[index]
        
        if self.transform:
            image = self.transform(image)
        if self.train == True:
            class_label = self.class_[index]
            return image, class_label
        else:
            return image

    def __len__(self):
        return len(self.img_paths)
    
class FaceCrop(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         self.device = torch.device('cpu') # train
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

    def __call__(self, img):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes,probs = self.mtcnn.detect(img)

        if not isinstance(boxes, np.ndarray):
            img = img[50:380, 80:300, :3]
        
        # boexes size 확인
        else:
            xmin = int(boxes[0, 0])-10
            ymin = int(boxes[0, 1])-10
            xmax = int(boxes[0, 2])+10
            ymax = int(boxes[0, 3])+10
            
            if xmin < 0: xmin = 0
            if ymin < 0: ymin = 0
            if xmax > 384: xmax = 384
            if ymax > 512: ymax = 512
            
            img = img[ymin:ymax, xmin:xmax, :3]
        
        img = Image.fromarray(img)
        
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + 'Face_crop'