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

from imageUtils import sliding_window,read_Color_Image,read_Binary_Mask, cleanUpMask, cleanUpMaskBlackPixels, boxesFromMask
from pathlib import Path
from collections import defaultdict

from PIL import Image

class CPDataset(Dataset):
    # Given a folder containing files stored following a certain regular expression,
    # Load all image files from the folder, put them into a list
    # At the same time, load all the labels from the folder names, put them into a list too!

    def __init__(self,dataFolder=None,transForm=None,listOfClasses=None,verbose = False):

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
        if verbose:
            self.classDictToFile("classDict.txt")

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

    def classDictToFile(self,outFile):
        """
        write the classDictionary to file

        """
        if (len(self.classesDict.items())>0):
            with open(outFile,"w") as f:
                for cod,unicode in self.classesDict.items():
                    f.write(str(unicode)+","+str(cod)+"\n")


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

# auxiliary functions for the ODDataset class
def build_masks(img, boxes, img_h, img_w):
    if not boxes: return np.zeros((0, img_h, img_w), dtype=np.uint8)
    masks = np.zeros((len(boxes), img_h, img_w), dtype=np.uint8)
    for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        masks[i, ymin:ymax, xmin:xmax] = (img[ymin:ymax, xmin:xmax] == 0)
    return masks

def load_boxes_from_label(label_path, img_w, img_h):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            _, cx, cy, w, h = map(float, line.strip().split())
            xmin, ymin = max(0, int((cx - w/2) * img_w)), max(0, int((cy - h/2) * img_h))
            xmax, ymax = min(img_w, int((cx + w/2) * img_w)), min(img_h, int((cy + h/2) * img_h))
            if xmax - xmin > 1 and ymax - ymin > 1: boxes.append([xmin, ymin, xmax, ymax])
    return boxes

class ODDataset(Dataset):
    """
        Dataset for object detection with
        pytorch predefined networks
    """
    def __init__(self, dataFolder = None, yoloFormat = True, slice = 500 , transform = None):
        """
            Receive a folder, should have an "images" and a "masks"
            subfolders with images with pre-defined names
            read all images and masks there, the images are pre-sliced

            Store all image and mask paths in two list of names
            store also the relation to the slices to the original images
            1) what was the original image 2) what coordinate in the original
            image is the 0,0 in the slice
        """
        # Data Structures:
        self.imageNameList = []
        self.maskNameList = []
        self.labelNameList = []
        self.transform = transform

        self.imageFolder = os.path.join(dataFolder,"images")
        self.maskFolder = os.path.join(dataFolder,"masks")
        self.labelFolder = os.path.join(dataFolder, "labels")

        self.slicesToImages = defaultdict(lambda:[])

        # create output Folder if it does not exist
        self.outFolder = os.path.join(dataFolder,"forOD")

        Path(self.outFolder).mkdir(parents=True, exist_ok=True)

        for dirpath, dnames, fnames in os.walk(self.maskFolder):
            for f in fnames:
                # read mask and image, everyone is binary
                mask = read_Binary_Mask(os.path.join(self.maskFolder,f))

                # image names are different if we read from original data or from already converted yoloFormat
                imageName = f[:-8]+f[-4:] if yoloFormat else f[2:-6]+".tif_resultat_noiseRemoval.tif"

                im = read_Binary_Mask(os.path.join(self.imageFolder,imageName))

                # Images need to be presliced beforehand.
                #print("Dataset: processing "+str(f))
                # images should already have the right size
                if im.shape[0] > slice or im.shape[1] > slice:raise Exception("ODDAtaset creator, wrongly presliced images")
                # no need to slice, but the images are tiles that come from other images, need to store this in self.slicesToImages[

                # Mask cleanup should happen when tile building
                #cleanUpMask(mask, areaTH = 200)
                #mask = cleanUpMaskBlackPixels(mask, im, areaTH = 150 )

                if np.sum(mask==0) > 100: # add only non-empty masks to the list of images and masks
                    #print(os.path.join(self.imageFolder,imageName)+" has boxes")

                    self.imageNameList.append( os.path.join(self.imageFolder,imageName) )
                    self.maskNameList.append( os.path.join(self.maskFolder,f) )
                    self.labelNameList.append(os.path.join(self.labelFolder, imageName[:-4] + ".txt"))

                # but add all tiles, even empty ones, to the dictionary
                self.slicesToImages[ imageName[:imageName.rfind("x")]+imageName[-4:]].append(imageName)


        #print(self.slicesToImages)
        #for k,v in self.slicesToImages.items():
        #    print(str(k)+" "+str(v))
        #print(len(self.slicesToImages))


    def __getitem__(self, idx):
        img = read_Binary_Mask(self.imageNameList[idx])
        #img_pil = Image.open(img_path).convert("RGB")
        img_h, img_w = img.shape[:2]
        boxes = load_boxes_from_label(self.labelNameList[idx], img_w, img_h)
        masks = build_masks(img, boxes, img_h, img_w)
        img_t = tv_tensors.Image(torch.as_tensor(img).unsqueeze(0).repeat(3, 1, 1))
        target = {"boxes": tv_tensors.BoundingBoxes(torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4), format="XYXY", canvas_size=(img_h, img_w)),
                "masks": tv_tensors.Mask(torch.as_tensor(masks, dtype=torch.uint8)),
                "labels": torch.ones((len(boxes),), dtype=torch.int64)}
        if self.transform is not None: img_t, target = self.transform(img_t, target)
        return img_t, target

    def __len__(self):
        return len(self.imageNameList)

    def getSliceFileInfo(self):
        """
        return the information about how everything was sliced
        """
        return self.slicesToImages
    


class ODDETRDataset(ODDataset):
    """
    Modified ODDataset that returns data in COCO-style format
    for use with HuggingFace DETR (DetrImageProcessor).
    
    CRITICAL: Returns images as PIL Images or numpy arrays,
    and annotations in absolute pixel coordinates [x, y, w, h].
    """

    def __getitem__(self, idx):
        img_path = self.imageNameList[idx]
        mask_path = self.maskNameList[idx]
        img = Image.open(img_path).convert("RGB")

        mask = np.array(Image.open(mask_path))
        numLabels, labelIm, stats, centroids = cv2.connectedComponentsWithStats(255 - mask)

        obj_ids = np.unique(labelIm)[1:]  # skip background
        masks = labelIm == obj_ids[:, None, None]

        annotations = []
        for i in range(len(obj_ids)):
            pos = np.nonzero(masks[i])
            if pos[0].size == 0 or pos[1].size == 0:
                continue  # skip empty masks
            
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            w, h = xmax - xmin + 1, ymax - ymin + 1  # +1 to include the boundary pixel

            # Skip degenerate boxes
            if w <= 1 or h <= 1:
                continue

            area = float(w * h)
            bbox = [float(xmin), float(ymin), float(w), float(h)]

            annotations.append({
                "bbox": bbox,
                "area": area,
                "category_id": 0,  # single class
                "iscrowd": 0
            })

        # FIXED: Don't add dummy annotations for empty tiles
        # The processor and training loop should handle empty annotation lists
        # If you must have something, the training code should filter these out
        
        # Return as numpy array (preferred by DetrImageProcessor)
        image = np.array(img)
        target = {"annotations": annotations}

        return image, target    


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
