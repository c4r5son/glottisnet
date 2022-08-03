from cgi import test
import numpy as np # linear algebra
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import gc
from model import UNet
import torch
from torchvision import transforms
import torch.nn.functional as F
import dataset
from torch.utils.data import DataLoader
import sys
import cv2
import time
gc.collect()
use_gpu = torch.cuda.is_available()

def thresh(x):
    if x == 0:
        return 0
    else:
        return 1

thresh = np.vectorize(thresh, otypes=[float])

def get_dataset(width_in, height_in,batch_size):
    #compose the all transformation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(), transforms.Resize((width_in,height_in))])

    #define the training dataset and the testing dataset
    training_data = dataset.ETIDataset(
    r'/sciclone/pscr/akurbach/Epiglottis_Data', train=True, transform=transform)

    test_data = dataset.ETIDataset(
    r'/sciclone/pscr/akurbach/Epiglottis_Data', train=False, transform=transform)

    #instantiate the dataloaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader

def train_step(inputs, labels, optimizer, criterion, unet, width_out, height_out):
    optimizer.zero_grad()
    
    #move inputs to gpu
    inputs = inputs.cuda()

    # forward + backward + optimize
    outputs = unet(inputs)
    # outputs.shape =(batch_size, n_classes, img_cols, img_rows)
    outputs = outputs.permute(0, 2, 3, 1)
    # outputs.shape =(batch_size, img_cols, img_rows, n_classes)
    m = outputs.shape[0]
    outputs = outputs.resize(m*width_out*height_out, 2)
    labels = labels.resize(m*width_out*height_out)
    labels = labels.long()

    #move labels to gpu
    labels = labels.cuda()

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss

def get_val_loss(test_dataloader, width_out, height_out, unet):
    val_loss = 0
    for X,y in test_dataloader:
        #move to gpu
        X = X.cuda()
        y = y.cuda()

        m = X.shape[0]

        outputs = unet(X)
        # outputs.shape =(batch_size, n_classes, img_cols, img_rows) 
        outputs = outputs.permute(0, 2, 3, 1)
        # outputs.shape =(batch_size, img_cols, img_rows, n_classes) 
        outputs = outputs.resize(m*width_out*height_out, 2)
        labels = y.resize(m*width_out*height_out)
        labels = labels.long()
        loss = F.cross_entropy(outputs, labels)
        val_loss += loss.data
    return val_loss

def train(unet, batch_size, epochs, epoch_lapse, threshold, learning_rate, criterion, optimizer, train_dataloader, test_dataloader,width_out, height_out):
    for t in range(epochs):
        total_loss = 0
        for batch, (X,y) in enumerate(train_dataloader):
            batch_loss = train_step(X , y, optimizer, criterion, unet, width_out, height_out)
            total_loss += batch_loss

        val_loss = get_val_loss(test_dataloader, width_out, height_out, unet)
        print("Total loss in epoch %i : %f and validation loss : %f" %(t+1, total_loss, val_loss))
    gc.collect()

def main():
    start = time.time()
    width_in = 284
    height_in = 284
    width_out = 196
    height_out = 196
    batch_size = 16
    train_dataloader, test_dataloader= get_dataset(width_in, height_in, batch_size)
    epochs = 64
    epoch_lapse = 50
    threshold = 0.5
    learning_rate = 0.001
    unet = UNet(in_channel=1,out_channel=2)
    if use_gpu:
        unet = unet.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unet.parameters(), lr = learning_rate, momentum=0.99)
    if sys.argv[1] == 'train':
        train(unet, batch_size, epochs, epoch_lapse, threshold, learning_rate, criterion, optimizer, train_dataloader, test_dataloader, width_out, height_out)
    
    PATH = "/sciclone/pscr/akurbach/models/unet.pth"
    torch.save(unet.state_dict(), PATH)
    print("Elapsed time:", time.time() - start)

if __name__ == "__main__":
    main()
