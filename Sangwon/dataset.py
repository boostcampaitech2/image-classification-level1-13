from torch.utils.data import Dataset
from PIL import Image

# Test,Train Dataset
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