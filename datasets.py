import numpy as np
import os
import sys
import re
from random import sample
import cv2

import torch
from torch.utils.data.dataset import Dataset
#from data_manipulation.datasets import get_slices_bb

class tDataset(Dataset):
    # Given a list of images, create a simple dataset with empty labels
    # this is for testing purposes, to have the dataset format
    # for lists of images

    def __init__(self,imageList,transform = None):
        # Data Structures:
        self.imageList = [np.moveaxis(im,-1,0) for im in  imageList]
        self.labelList = ["nolabel"]*len(imageList) # a list to store our labels
        self.transform = transform
        #print(self.transform)

    def __getitem__(self, index):
            # We will need to transform our images to torch tensors
            currentImage = torch.from_numpy(self.imageList[index].astype(np.float32)) # transform to torch tensor with floats
            if self.transform :
                currentImage = self.transform(currentImage) # apply transforms that may be necessary to
            inputs = currentImage
            return inputs, self.labelList[index]
    def __len__(self):
        return len(self.imageList)


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

                    # Read the file as an opencv image
                    currentImage = cv2.imread(os.path.join(root,f))
                    if currentImage is None: raise Exception("CPDataset Constructor, problems reading file "+f)

                    # Now be careful, pytorch needs the pixel dimension at the start,
                    # so we have to change the way the images are stored (moveaxis)
                    # We also store the image in the image list
                    currentImage = np.moveaxis(currentImage,-1,0)
                    currentImage = currentImage[:,:,::-1] #change from BGR to RGB
                    self.imageList.append(currentImage)

                    # Finally, maintain a class dictionary, the codes of the
                    # class are assigned in the order in which we encounter them
                    # also, store the label in the label list
                    if currentClass not in self.classesDict: self.classesDict[currentClass] = len(self.classesDict)
                    self.labelList.append(self.classesDict[currentClass])

    def __getitem__(self, index):

            # When we call getitem, we will generally be passing a piece of data to pytorch
            # First, simply retrieve the proper image from the list of images
            currentImage = self.imageList[index]
            # We will need to transform our images to torch tensors
            currentImage = torch.from_numpy(currentImage.astype(np.float32)) # transform to torch tensor with floats
            if self.transform :
                currentImage = self.transform(currentImage) # applly transforms that may be necessary to

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


if __name__ == '__main__':
    da=CPDataset(sys.argv[1])

    tr,vd=da.breakTrainValid(0.4)

    print(len(tr))
    print(len(vd))

    print(vd.numClasses())
    #print(vd.classesDict)

    with open("classDict.txt","w") as f:
        for cod,unicode in vd.classesDict.items():
            f.write(str(unicode)+","+str(cod)+"\n")

#    for x,y in tr:
        #print(x)
#        print(y)
