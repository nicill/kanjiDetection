import numpy as np
import os
import sys
import re
from random import sample
import cv2

import torch
from torch.utils.data.dataset import Dataset

from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from imageUtils import sliding_window,read_Color_Image,read_Binary_Mask, cleanUpMask
from buildTrainValidation import boxesFromMask
from pathlib import Path
from collections import defaultdict

from PIL import Image

class CPDataset(Dataset):
    # Given a folder containing files stored following a certain regular expression,
    # Load all image files from the folder, put them into a list
    # At the same time, load all the labels from the folder names, put them into a list too!

    def __init__(self,dataFolder=None,transForm=None,listOfClasses=None):

        # Data Structures:
        self.classesDict = {} #a dictionary to store class codes
        self.imageList = [] # a list to store our images
        self.labelList = [] # a list to store our labels
        self.transform = transForm

        # Use this if so we can also build "empty DataSets"
        if dataFolder is not None:
            # Use os.walk to obtain all the files in our folder
            for root, dirs, files in os.walk(dataFolder):
                #Traverse all files
                for f in files:
                    # For every file, get its category
                    currentClass = root.split(os.sep)[-1]

                    # Read the file as a grayscale opencv image
                    currentImage = cv2.imread(os.path.join(root,f),0)
                    if currentImage is None: raise Exception("CPDataset Constructor, problems reading file "+f)

                    # now binarize strictly
                    currentImage[currentImage<=100] = 0
                    currentImage[currentImage>100] = 255

                    # Now be careful, pytorch needs the pixel dimension at the start,
                    # so we have to change the way the images are stored (moveaxis)
                    # We also store the image in the image list
                    self.imageList.append(currentImage)

                    # Finally, maintain a class dictionary, the codes of the
                    # class are assigned in the order in which we encounter them
                    # also, store the label in the label list
                    if currentClass not in self.classesDict: self.classesDict[currentClass] = len(self.classesDict)
                    self.labelList.append(self.classesDict[currentClass])

    def __getitem__(self, index):

            # When we call getitem, we will generally be passing a piece of data to pytorch
            # First, simply retrieve the proper image from the list of images
            currentImage = cv2.cvtColor(self.imageList[index],cv2.COLOR_GRAY2RGB)
            currentImage = np.moveaxis(currentImage,-1,0)
            #currentImage = currentImage[:,:,::-1] #change from BGR to RGB

            # We will need to transform our images to torch tensors
            currentImage = torch.from_numpy(currentImage.astype(np.float32)) # transform to torch tensor with floats
            if self.transform :
                currentImage = self.transform(currentImage) # apply transforms that may be necessary to

            inputs = currentImage

            # in this case, and to make things simple, the target is the code of the class the patch belongs to
            target = self.labelList[index]

            return inputs, target

    def __len__(self):
        return len(self.imageList)

    def numClasses(self):return len(np.unique(list(self.classesDict.keys())))

    #Create two dataset (training and validation) from an existing one, do a random split
    def breakTrainValid(self,proportion):
        train=CPDataset(None)
        valid=CPDataset(None)
        train.classesDict = self.classesDict
        valid.classesDict = self.classesDict

        train.transform = self.transform
        valid.transform = self.transform

        #randomly shuffle the data
        toDivide = sample(list(zip(self.imageList,self.labelList)),len(self.imageList))

        for i in range(int(len(self)*proportion)):
            valid.imageList.append(toDivide[i][0].copy())
            valid.labelList.append(toDivide[i][1])

        for i in range(int(len(self)*proportion),len(self)):
            train.imageList.append(toDivide[i][0].copy())
            train.labelList.append(toDivide[i][1])

        return train,valid

class tDataset(CPDataset):
    # Given a list of images, create a simple dataset with empty labels
    # this is for testing purposes, to have the dataset format
    # for lists of binarized grayscale images

    def __init__(self,imageList,transform = None):
        # Data Structures:
        self.imageList = imageList
        self.labelList = ["nolabel"]*len(imageList) # a list to store our labels
        self.transform = transform
        #print(self.transform)
# sources
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# https://pytorch.org/vision/main/models.html
class ODDataset(Dataset):
    """
        Dataset for object detection with
        pytorch predefined networks
    """
    def __init__(self, dataFolder = None, slice = 500 , transform = None):
        """
            Receive a folder, should have an "images" and a "masks"
            subfolders with images with pre-defined names
            read all images and masks there, slice them
            with the size given and create a new folder "forOD"
            with the cut masks and images.

            Store all image and mask paths in two list of names
            store also the relation to the slices to the original images
            1) what was the original image 2) what coordinate in the original
            image is the 0,0 in the slice
        """

        # Data Structures:
        self.imageNameList = []
        self.maskNameList = []
        self.transform = transform

        self.imageFolder = os.path.join(dataFolder,"images")
        self.maskFolder = os.path.join(dataFolder,"masks")
        self.slicesToImages = defaultdict(lambda:[])

        # create output Folder if it does not exist
        self.outFolder = os.path.join(dataFolder,"forOD")

        Path(self.outFolder).mkdir(parents=True, exist_ok=True)

        for dirpath, dnames, fnames in os.walk(self.maskFolder):
            for f in fnames:
                # read mask and image, everyone is binary
                mask = read_Binary_Mask(os.path.join(self.maskFolder,f))
                imageName = f[2:-6]+".tif_resultat_noiseRemoval.tif"
                im = read_Binary_Mask(os.path.join(self.imageFolder,imageName))

                # slice mask and image together, store them in the forOD folder
                wSize = (slice,slice)
                count = 0
                for (x, y, window) in sliding_window(im, stepSize = int(slice*0.8), windowSize = wSize ):
                    # get mask window
                    if window.shape == (slice,slice) :
                        # do we need to fix this?
                        # this should be done better, with padding!, maybe https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html
                        maskW = mask[y:y + wSize[1], x:x + wSize[0]]
                        # discard empty masks
                        cleanUpMask(maskW)
                        if np.sum(maskW==0) > 100:
                            # store them both
                            cv2.imwrite(os.path.join(self.outFolder,"Tile"+str(count)+f[2:-6]+".png"),window)
                            self.imageNameList.append( os.path.join(self.outFolder,"Tile"+str(count)+f[2:-6]+".png") )
                            cv2.imwrite(os.path.join(self.outFolder,"MaskTile"+str(count)+f[2:-6]+".png"),maskW)
                            self.maskNameList.append(os.path.join(self.outFolder,"MaskTile"+str(count)+f[2:-6]+".png"))

                            self.slicesToImages[imageName].append(("Tile"+str(count)+f[2:-6]+".png",x,y))

                            count+=1
        #print(slicesToImages)

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imageNameList[idx]
        mask_path = self.maskNameList[idx]
        img = Image.open(img_path).convert("RGB")
        #img = read_Color_Image(img_path)

        #mask = read_Binary_Mask(mask_path)
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        numLabels, labelIm,  stats, centroids = cv2.connectedComponentsWithStats(255-mask)

        obj_ids = np.unique(labelIm)
        #print(obj_ids)
        #obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split into a set of masks
        masks = labelIm == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmax-xmin > 1 and ymax-ymin >1:
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                print("weird box "+str([xmin, ymin, xmax, ymax]))
                raise Exception("dataset.py, wrong mask should have been cleaned up ")

        # only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(torch.as_tensor(boxes), format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask( masks )
        target["labels"] = torch.as_tensor(labels)

        if self.transform is not None:
            img, target = self.transform(img, target)

        #print(target)
        return img, target

    def __len__(self):
        return len(self.imageNameList)

    def getSliceFileInfo(self):
        """
        return the information about how everything was sliced
        """
        return self.slicesToImages


if __name__ == '__main__':
    print("This main does nothing at the moment, why are you calling it?")
    """
    da=CPDataset(sys.argv[1])

    tr,vd=da.breakTrainValid(0.4)

    print(len(tr))
    print(len(vd))

    print(vd.numClasses())
    #print(vd.classesDict)

    with open("classDict.txt","w") as f:
        for cod,unicode in vd.classesDict.items():
            f.write(str(unicode)+","+str(cod)+"\n")
    """
#    for x,y in tr:
        #print(x)
#        print(y)
