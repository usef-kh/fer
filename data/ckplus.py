import collections
import os

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.dataset import CustomDataset


def load_data(path='datasets/ckplus'):
    """ prepare all image paths and their corresponding emotions. Group all images by subject.

        the dataset is slightly odd such that each subject has ~3 identical images.
        So, we will be grouping these images together such that they end up in the same dataset split.
        This way we ensure no overlap between train, validation, and testing data.
        If we dont, we would have ~100% training, validation, and testing accuracy.

        input: path to ckplus dir
        output: dict key: emotion, value: dict(key: subject, value: list of image paths) """

    ckplus = {}
    for dir, subdir, files in os.walk(path):

        if files:
            sections = os.path.split(dir)
            emotion = sections[-1]

            # create child dictionary that groups images of the same subject id together
            subjects = collections.defaultdict(list)
            for file in files:
                # the subject id is present at the beginning of the file name
                subject = file.split("_")[0]
                subjects[subject].append(os.path.join(dir, file))

            ckplus[emotion] = subjects

    return ckplus


def prepare_data(data):
    """ Prepare data for modeling
        input: dict of key: emotion (str), values: image paths [list]
        output: image and label array """

    # count number of images
    n_images = sum(len(paths) for paths in data.values())
    emotion_mapping = {emotion: i for i, emotion in enumerate(data.keys())}

    # allocate required arrays
    image_array = np.zeros(shape=(n_images, 48, 48))
    image_label = np.zeros(n_images)

    i = 0
    for emotion, img_paths in data.items():
        for path in img_paths:
            image_array[i] = np.array(Image.open(path))  # load image from path into numpy array
            image_label[i] = emotion_mapping[emotion]  # convert emotion to its emotion id
            i += 1

    return image_array, image_label


def split_data(data):
    """ Train, val, test, split by subject
        input: dict key: emotion, value: dict(key: subject, value: list of image paths)
        output: 3 dictionaries of key: emotion, value: list of image paths """

    train = collections.defaultdict(list)
    test = collections.defaultdict(list)
    val = collections.defaultdict(list)

    for emotion, subjects in data.items():

        # shuffle each emotion's subjects and split them using a 0.8, 0.2, 0.2 split
        subjects_train, subjects_test = train_test_split(list(subjects.keys()), test_size=0.2, random_state=1, shuffle=True)
        subjects_train, subjects_val = train_test_split(subjects_train, test_size=0.25, random_state=1, shuffle=True)  # 0.25 * 0.8 = 0.2

        for subject in subjects_train:
            train[emotion].extend(subjects[subject])

        for subject in subjects_val:
            val[emotion].extend(subjects[subject])

        for subject in subjects_test:
            test[emotion].extend(subjects[subject])

    return train, val, test


def get_dataloaders(path='datasets/ckplus'):
    ckplus = load_data(path)
    train, val, test = split_data(ckplus)

    xtrain, ytrain = prepare_data(train)
    xval, yval = prepare_data(val)
    xtest, ytest = prepare_data(test)

    mu, st = 0, 1
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
        transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(mu,), std=(st,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(mu,), std=(st,))
    ])

    train = CustomDataset(xtrain, ytrain, train_transform)
    val = CustomDataset(xval, yval, test_transform)
    test = CustomDataset(xtest, ytest, test_transform)

    trainloader = DataLoader(train, batch_size=100, shuffle=True, num_workers=2)
    valloader = DataLoader(val, batch_size=100, shuffle=True, num_workers=2)
    testloader = DataLoader(test, batch_size=100, shuffle=True, num_workers=2)

    return trainloader, valloader, testloader


if __name__ == '__main__':
    ckplus = load_data('../datasets/ckplus')

    train, val, test = split_data(ckplus)

    # for k, v in train.items():
    #     print(k)
    #     print(*v, sep='\n')

    image_array, image_label = prepare_data(train)
    #
    #
    # train, val, test = split_data(image_array, image_label)
    #
    for i in range(12):
        print(image_label[i])
        plt.figure()
        plt.imshow(image_array[i])
        plt.show()

    # print(ckplus)
