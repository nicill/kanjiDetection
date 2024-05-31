"""
File to build training and Validation datasets
For Kanji Detection using deep learning
"""
from imageUtils import sliding_window,read_Color_Image,read_Binary_Mask
import os
import cv2
import sys
from random import randint
from pathlib import Path

def boxesFromMask(img, cl = 0, yoloFormat = True):
    """
    Return a list of box coordinates
    From a binary image

    Parameters
    ----------
    im : ndarray
        taget image

    Returns
    -------
    list: list of box coordinater
    """
    # void small regions
    threshold = 500

    # Binarize, just in case
    img[img<=10] = 0
    img[img>10] = 255
    _, _, stats, centroids = cv2.connectedComponentsWithStats(255-img)

    out = []
    # Avoid first centroid, unbounded component
    for j in range(1,len(centroids)):
        if yoloFormat:
            if stats[j][4] > threshold:
                h, w = img.shape
                bw = stats[j][2]
                bh = stats[j][3]
                cx = stats[j][0] + bw/2
                cy = stats[j][1] + bh/2
                # normalize and append
                out.append((cl,cx/w,cy/h,bw/w,bh/h))
        else:
            raise Exception("UNSUPORTED BOX FORMAT")
    return out

def sliceAndBox(im,mask,slice):
    """
    Given image and mask, slice them
    output image slices and mask slice
    and text files
    """
    out = []
    wSize = (slice,slice)
    for (x, y, window) in sliding_window(im, stepSize = slice, windowSize = wSize ):
        boxList = []
        # get mask window
        maskW = mask[y:y + wSize[1], x:x + wSize[0]]
        # The mask was already binarized
        # compute box coords
        coords = boxesFromMask(maskW)
        # add window, mask window and boxlist
        out.append(("x"+str(x)+"y"+str(y),window,maskW,coords))
    return out

def boxCoordsToFile(file,boxC):
    """
        Receive a list of tuples
        with bounding boxes
        and write it to file

    """
    def writeTuple(tup):
        c,px,py,w,h = tup
        f.write(str(c)+" "+str(px)+" "+str(py)+" "+str(w)+" "+str(h)+"\n")

    with open(file, 'a') as f:
        list(map( writeTuple, boxC))

def buildTrainValid(imageFolder, maskFolder, slice, outTrain, outVal, perc):
    """
        Receives a folder with images
        And another of masks
        CAREFUL! name correspondences between
        mask and image files
    """
    # create output directories if they do not exist
    #for d in [os.path.join(outTrain,"images"),os.path.join(outTrain,"masks"),
    #os.path.join(outTrain,"labels"),os.path.join(outVal,"images"),
    #os.path.join(outVal,"masks"),os.path.join(outVal,"labels")]:
    #    Path(d).mkdir(parents=True, exist_ok=True)
    for d in [os.path.join(outTrain,"images"),os.path.join(outTrain,"labels"),
    os.path.join(outVal,"images"),os.path.join(outVal,"labels")]:
        Path(d).mkdir(parents=True, exist_ok=True)

    for dirpath, dnames, fnames in os.walk(maskFolder):
        for f in fnames:
            # read mask and image, everyone is binary
            mask = read_Binary_Mask(os.path.join(maskFolder,f))
            imageName = f[2:-6]+".tif_resultat_noiseRemoval.tif"
            im = read_Binary_Mask(os.path.join(imageFolder,imageName))

            # Masks are not the perfect size, reshape
            mask = cv2.resize(mask, (im.shape[1],im.shape[0]))

            # binarize mask
            mask[mask<10] = 0
            mask[mask>10] = 255

            # call slice and box
            sIML = sliceAndBox(im,mask,slice)

            # store results (make up names)
            for suffix,i,m,l in sIML:
                newFileName = f[2:-6]+suffix

                # randomDraw train/val
                outDir = outTrain if randint(1,100) < perc else outVal
                # store files including text file
                cv2.imwrite(os.path.join(outDir,"images",newFileName+".png"),i)
                #cv2.imwrite(os.path.join(outDir,"masks",newFileName+"MASK.png"),m)
                boxCoordsToFile(os.path.join(outDir,"labels",newFileName+".txt"),l)

def buildTesting(imageFolder, maskFolder, outTest):
    """
        Receives a folder with images and another with masks
        and copies to a "testing" file, those with no mask
        (name correspondences between
        mask and image files )
    """
    Path(os.path.join(outTest,"images")).mkdir(parents=True, exist_ok=True)

    for dirpath, dnames, fnames in os.walk(imageFolder):
        for f in fnames:
            # for each image, check if it has been noise removed
            # and if it has a mask
            if "noiseRemoval" in str(f):
                maskName = "KP"+str(f[:-30])+"AN.jpg"
                if not Path(os.path.join(maskFolder,maskName)).is_file():
                    im = cv2.imread(os.path.join(imageFolder,f))
                    cv2.imwrite(os.path.join(outTest,"images",f),im)
