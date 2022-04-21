import model

import torch
from torchvision import transforms

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

class glottisnet():
    def __init__(self, weights_path="models/unet_dummy.pth"):
        #compose the all transformation
        self.image_transforms = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(), transforms.Resize((284,284))])

        #load weights to network
        self.weights_path = weights_path
        self.device = "cpu"

        self.unet = model.UNet(in_channel=1, out_channel=2)
        self.unet.to(self.device)
        self.unet.load_state_dict(torch.load(weights_path, map_location=self.device))

    def __predict(self, model, img, device) -> np.ndarray:
        '''This function outputs a prediction mask from the loaded model and image. Not for public use.
        @param model the model to use for prediction
        @param img the image to predict on
        @param device cpu or CUDA'''
        model.eval()
        with torch.no_grad():
            images = img.to(device)
            output = model(images)

            predicted_masks = (output.squeeze() >= 4.3).float().cpu().numpy()
        return(predicted_masks)

    def get_mask(self, img) -> np.ndarray:
        '''This method returns a numpy array mask based on the passed image path.
        @param img an image to get a mask for
        
        @ret mask'''
        original_height, original_width = tuple(img.shape[:2])
        
        image_transformed = self.image_transforms(img)
        image_transformed = image_transformed.unsqueeze(0)
        
        image_mask = self.__predict(self.unet, image_transformed, self.device)
        image_mask = cv2.resize(image_mask[0], (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            
        return(image_mask)

    def plot_example(self,image_path,threshold=90000):
        '''This method plots a given image and its mask when run through the network with matplotlib.
        @param image_path path to an example image
        @param threshold the area threshold to consider a glottis found'''
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = self.get_mask(image)

        _, (axis1, axis2) = plt.subplots(2)

        box = glottisnet.__boundingBox(mask)
        #area = (xmax-xmin)*(ymax-ymin)
        area = (box[1][0]-box[0][0])*(box[1][1]-box[0][1])

        #if the area of the box is greater than the threshold value draw it
        if area >= threshold:
            color = (255, 0, 0)
            thickness = 5
            cv2.rectangle(image,box[0],box[1],color,thickness)

        axis1.imshow(image)
        axis2.imshow(mask)
        plt.show()

    def getBoundingBox(self, img, threshold=90000):
        '''This method takes an image and returns a bounding box if the area of the box is greater than the threshold, otherwise it returns None.
        @param img an image to process
        @param threshold the area threshold to consider a glottis found
        @ret a bounding box of the form ((xmin, ymin), (xmax, ymax))'''
        mask = self.get_mask(img)

        box = glottisnet.__boundingBox(mask)
        #area = (xmax-xmin)*(ymax-ymin)
        area = (box[1][0]-box[0][0])*(box[1][1]-box[0][1])

        #if the area of the box is greater than the threshold value draw it
        if area >= threshold:
            return box
        else:
            return None

    def drawBoundingBox(self, img, threshold=90000):
        '''This method takes an image and returns an image with a bounding box drawn on it if there is one.
        @param img an image to process
        @param threshold the area threshold to consider a glottis found
        @ret an image with bounding box'''
        mask = self.get_mask(img)

        box = glottisnet.__boundingBox(mask)
        #area = (xmax-xmin)*(ymax-ymin)
        area = (box[1][0]-box[0][0])*(box[1][1]-box[0][1])

        #if the area of the box is greater than the threshold value draw it
        if area >= threshold:
            color = (255, 0, 0)
            thickness = 5
            cv2.rectangle(img,box[0],box[1],color,thickness)
            return img
        else:
            return img
    
    @classmethod
    def __boundingBox(self, arr):
        '''This method gives a bounding box around the highlighted values of an inputted numpy array.
        @param arr Image
        @ret (xmin, ymin), (xmax, ymax)
        '''
        rows = np.any(arr, axis=1)
        cols = np.any(arr, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        return (xmin, ymin), (xmax, ymax)

def parse_args() -> argparse.Namespace:

    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Glottisnet Command line interface. Specify an image to run through and plot.")
    parser.add_argument(
        "-i",
        "--input",
        help=(
            "Path to image to run through Glottisnet"
        ),
        required=True
    )
    parser.add_argument(
        "--model",
        choices=["dummy","real"],
        help=(
            "Select the model you want to use. Options \"dummy\" or \"real\"."
        )
    )
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()

    if args.model == "real":
        #load the real weights model
        glottisnet = glottisnet(weights_path="models/unet_real.pth")
    else:
        #if nothing specified or dummy use the weights trained with images from intubation manikin
        glottisnet = glottisnet()

    glottisnet.plot_example(args.input)
    
