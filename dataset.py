import torch
import torchvision.transforms as transforms # для преобразований тензоров
import os
import cv2
from pathlib import Path


class MakeDataset(torch.utils.data.Dataset):
    def __init__(self, dataRootFolder, image_size, mode='train'):
        super().__init__()
        assert mode in ['train', 'test']
        self.mode = mode
        self.dataRootFolder = dataRootFolder
        self.files = os.listdir(Path(dataRootFolder) / self.mode)
        self.labels = [self.get_class_name(filename) for filename in self.files]
        self.classes = list(set(self.labels))
        self.labels = list(map(lambda x: self.classes.index(x), self.labels))
        self.len_ = len(self.files)
        self.image_size = image_size
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize(self.image_size)])

    def __len__(self):  # функция определения размера датасета
        return self.len_

    def get_class_name(self, filename):
        return filename.split('type')[1].split('_')[1]

    def __getitem__(self, index):  # функция чтения и обработки каждой картинки
        image = cv2.imread(str(Path(self.dataRootFolder) / self.mode / self.files[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        aug_image = self.transform(image)
        label = self.labels[index]
        return aug_image, label