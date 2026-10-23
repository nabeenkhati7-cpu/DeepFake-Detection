import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2 
import numpy as np

import uuid
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate

from transforms.albu import IsotropicResize

class DeepFakesDataset(Dataset):
    def __init__(self, images, labels, image_size, mode='train'):
        self.x = images
        self.y = torch.as_tensor(labels, dtype=torch.long)
        self.image_size = image_size
        self.mode = mode
        self.n_samples = len(images)

        if self.mode == 'train':
            self.transform = self.create_train_transforms(self.image_size)
        else:
            self.transform = self.create_val_transform(self.image_size)
    
    def create_train_transforms(self, size):
        return Compose([
            ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
            GaussNoise(p=0.3),
            #GaussianBlur(blur_limit=3, p=0.05),
            HorizontalFlip(),
            OneOf([
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.4),
            ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]
        )
        
    def create_val_transform(self, size):
        return Compose([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ])

    

    def __getitem__(self, index):
        image_path = self.x[index]

        image = cv2.imread(str(image_path))

        if image is None:
            from PIL import Image

            try:
                with Image.open(str(image_path)) as pil_img:
                    image = np.array(pil_img.convert("RGB"))
            except Exception as exc:
                raise RuntimeError(
                    f"Could not read image: {image_path}"
                ) from exc
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image=image)['image']

        return torch.tensor(image, dtype=torch.float32), self.y[index]


    def __len__(self):
        return self.n_samples

 
