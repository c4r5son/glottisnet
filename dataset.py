import glob
import os
from PIL import Image

from torch.utils import data
from torchvision.datasets.folder import pil_loader

from torchvision import transforms

import torch.nn.functional as F
from torch import from_numpy
import numpy as np

class ETIDataset(data.Dataset):
    '''This dataset returns and Image and a bit mask both as Images or as specified by the transformation.'''

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = None

        self.prepare()

    def prepare(self):
        self.samples = []

        if self.train:
            #add all of the file in the training mask folder to a list
            label_path = glob.glob(
                os.path.join(self.root, 'training_masks/*.jpg'))

            #add all of the file in the trainging image folder to a list
            image_paths = glob.glob(
                os.path.join(self.root, 'training_images/*.jpg'))
        else:
            #add all of the file in the testing mask folder to a list
            label_path = glob.glob(
                os.path.join(self.root, 'testing_masks/*.jpg'))
            #add all of the file in the testing image folder to a list
            image_paths = glob.glob(
                os.path.join(self.root, 'testing_images/*.jpg'))

        for image_path in image_paths:
            #add both the label_path and the image_path to the sample list
            if os.path.exists(image_path):
                self.samples.append((image_path, label_path))
            else:
                raise FileNotFoundError

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]

        image = pil_loader(image_path)
        label = None
        
        #find the right label to match the image
        for mask in label_path:
            if mask.rsplit('/')[-1] == image_path.rsplit('/')[-1]:
                label = mask
                break
        
        #load the label as an image
        label = pil_loader(label)

        if self.transform is not None:
            #not sure whether or not to transform the label
            image = self.transform(image)
            label = self.transform(label)
            labelTransform = transforms.Resize((196,196))
            label = labelTransform(label)
            

        return image, label

    def __len__(self):
        return len(self.samples)


def main():
    transform = transforms.Compose(
        [transforms.Resize(64), transforms.ToTensor()])
    #must supply your path to the data
    loader = data.DataLoader(
        ETIDataset(r'/Users/carsonstillman/data/Epiglottis_Data/', transform=transform),
        batch_size=1,
        shuffle=True)

    count = 0
    for i, (x, y) in enumerate(loader):
        count+=1
        print(x.size())
        print(y.size())
        if count>=1:
            break


if __name__ == '__main__':
    main()
