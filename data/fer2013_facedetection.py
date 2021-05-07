import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from data.dataset import CustomDataset

import cv2


def load_data(path='datasets/fer2013/fer2013.csv'):
    fer2013 = pd.read_csv(path)
    emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    return fer2013, emotion_mapping


def face_detection(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.05, 1)

    if faces != ():

        x, y, w, h = faces[0]  # theres only 1 face in our images

        face = frame[y:y + h, x:x + w]  # Extract face from frame
        resized_face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)

        return resized_face

    else:
        return frame


def prepare_data(data):
    """ Prepare data for modeling
        input: data frame with labels und pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))

        image = face_detection(image.astype(np.uint8))

        image_array[i] = image

    return image_array, image_label


def get_dataloaders():
    fer2013, emotion_mapping = load_data()

    xtrain, ytrain = prepare_data(fer2013[fer2013['Usage'] == 'Training'])
    xval, yval = prepare_data(fer2013[fer2013['Usage'] == 'PrivateTest'])
    xtest, ytest = prepare_data(fer2013[fer2013['Usage'] == 'PublicTest'])

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
    fer2013, emotion_mapping = load_data('../datasets/fer2013/fer2013.csv')

    xtrain, ytrain = prepare_data(fer2013[fer2013['Usage'] == 'Training'])

    for i in range(50):

        try:
            resized = face_detection(xtrain[i].astype(np.uint8))
            fig, axes = plt.subplots(2, 1)
            for ax, img in zip(axes, (xtrain[i], resized)):
                ax.imshow(img)
            plt.show()

        except Exception as e:
            plt.figure()
            plt.imshow(xtrain[i])
            plt.show()
    #
    # for i in range(12):
    #     print(xtrain[i])
    #     plt.sub()
    #     plt.imshow(xtrain[i])
    #     plt.show()
