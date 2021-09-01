
from torchvision.transforms.functional import crop

def crop800(image):
    return crop(image, 80, 50, 370, 300)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

from torchvision import transforms
import torchvision.datasets
from torchvision.transforms import Resize, ToTensor, CenterCrop, Normalize, GaussianBlur, RandomRotation,RandomResizedCrop, ColorJitter, RandomHorizontalFlip

def get_transforms(need=('train', 'val'), img_size=(512, 384)):
    transformations = {}
    if 'train' in need:
        transformations['train'] = transforms.Compose([
            Resize((img_size[0], img_size[1])),
#             RandomResizedCrop((350,250),scale=(0.5,1.0)),
            RandomHorizontalFlip(),
            RandomRotation([-2, +2]),
            GaussianBlur(51, (0.1, 2.0)),
            ColorJitter(brightness=0.7, saturation=0.7, hue=0.5),  # todo : param
            ToTensor(),
            transforms.Lambda(crop800),
            Normalize(mean, std),
        ])
    if 'val' in need:
        transformations['val'] = transforms.Compose([
            Resize((img_size[0], img_size[1])),
            CenterCrop(350),
            ToTensor(),
            Normalize(mean, std),
        ])
    return transformations



from torch.utils.data import Dataset
from PIL import Image

class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1.jpg": MaskLabels.mask,
        "mask2.jpg": MaskLabels.mask,
        "mask3.jpg": MaskLabels.mask,
        "mask4.jpg": MaskLabels.mask,
        "mask5.jpg": MaskLabels.mask,
        "incorrect_mask.jpg": MaskLabels.incorrect,
        "normal.jpg": MaskLabels.normal
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, img_dir, transform=None):
        """
        MaskBaseDataset을 initialize 합니다.

        Args:
            img_dir: 학습 이미지 폴더의 root directory 입니다.
            transform: Augmentation을 하는 함수입니다.
        """
        self.img_dir = img_dir

        self.setup()

    def set_transform(self, transform):
        """
        transform 함수를 설정하는 함수입니다.
        """
        self.transform = transform
        
    def setup(self):
        """
        image의 경로와 각 이미지들의 label을 계산하여 저장해두는 함수입니다.
        """
        profiles = os.listdir(self.img_dir)
        for profile in profiles:
            for file_name, label in self._file_names.items():
                img_path = os.path.join(self.img_dir, profile, file_name) 
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.mask_labels.append(label)

                    id, gender, race, age = profile.split("_")
                    gender_label = getattr(GenderLabels, gender)
                    age_label = AgeGroup.map_label(age)

                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

    def __getitem__(self, index):
        """
        데이터를 불러오는 함수입니다. 
        데이터셋 class에 데이터 정보가 저장되어 있고, index를 통해 해당 위치에 있는 데이터 정보를 불러옵니다.
        
        Args:
            index: 불러올 데이터의 인덱스값입니다.
        """
        # 이미지를 불러옵니다.
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        
        # 레이블을 불러옵니다.
        mask_label = self.mask_labels[index]
        gender_label = self.gender_labels[index]
        age_label = self.age_labels[index]
        multi_class_label = mask_label * 6 + gender_label * 3 + age_label
        
        # 이미지를 Augmentation 시킵니다.
        image_transform = self.transform(image)
        return image_transform, multi_class_label
        
        if not self.split_labels:
            multi_class_label = mask_label * 6 + gender_label * 3 + age_label
            return image_transform, multi_class_label
        else:
            return image_transform, (mask_label, gender_label, age_label)

    def __len__(self):
        return len(self.image_paths)