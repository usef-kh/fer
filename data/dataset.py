import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = np.array(self.images[idx])


        # image =  image.squeeze(0)
        # print(image.shape)
        img = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
        # img = np.uint8(img)#cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
        img = cv2.equalizeHist(img)
        
        # img = self.clahe.apply(img)
        img = Image.fromarray(img)


        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx]).type(torch.long)
        sample = (img, label)

        return sample
