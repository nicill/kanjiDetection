import numpy as np
import cv2
import sys
from math import sqrt
import os
from pathlib import Path

def read_Color_Image(path):
    #Read tif image or png image
    fileFormat = path[-3:]
    if fileFormat in ["png","jpg"]:
      retVal = cv2.imread(path,cv2.IMREAD_COLOR)
    elif fileFormat == "tif":
      retVal = cv2.imread(path,cv2.IMREAD_UNCHANGED)
      # As the shape is x*y*4 and not x*y*3, adapt
      retVal = retVal[:,:,:3]
    if retVal is None: raise Exception("Reading Color image, something went wrong with the file name "+str(path))
    return retVal

#Read binary image
def read_Binary_Mask(path):

    retVal = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if retVal is None: raise Exception("Reading GRAYSCALE image, something went wrong with the file name "+str(path))

    # Binarize
    retVal[retVal<=50] = 0
    retVal[retVal>50] = 255
    return retVal

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def distPoints(p,q):
    """
    Euclidean Distance bewteen 2D points
    """
    return sqrt( (p[0]-q[0])*(p[0]-q[0])+(p[1]-q[1])*(p[1]-q[1]))

def predictionsToKanjiImages(im,mask,path,imCode,storeContext=False):
    """
    Function that receives a prediction
    Binary mask and an image and stores the
    resulting Kanji images in the hard drive
    Adds a boolean flag to also store context images
    """
    def processComponent(l):
        """
        Inner function to store
        the part marked by the label
        image to disk
        """
        #nonlocal im
        x = stats[l][0]
        y = stats[l][1]
        w = stats[l][2]
        h = stats[l][3]
        subIm = im[y:y+h,x:x+w]
        cv2.imwrite(os.path.join(path,
                    imCode+"kanjiX"+str(x)+"Y"+str(y)+"H"+str(h)+"W"+str(w)+".jpg")
                    ,subIm)
        if storeContext:

            newIm = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
            newSubIm = newIm[y:y+h,x:x+w]
            newSubIm[subIm==0] = (0,0,255)

            subImC = newIm[max(0,int(y-contextSize/2)):y+h+contextSize,max(0,int(x-contextSize)):x+w+contextSize]
            cv2.imwrite(os.path.join(path,
                        "CONTEXT"+imCode+"kanjiX"+str(x)+"Y"+str(y)+"H"+str(h)+"W"+str(w)+".jpg")
                        ,subImC)

    # create output folder if necessary
    Path(path).mkdir(parents=True, exist_ok=True)

    # Threshold  the image to make sure it is binary
    contextSize = 500
    strictBinarization(mask)
    #compute connected components
    numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(255-mask)
    #traverse all labels but ignore label 0 as it contains the background
    list(map(processComponent,range(1,numLabels)))

def cleanUpMask(mask, areaTH = 100, thicknessTH = 20):
    """
    Receive a Mask with the position of kanji
    erase regions that are too small or not fat enough
    """
    # Binarize, just in case
    mask[mask<=10] = 0
    mask[mask>10] = 255
    numLabels, labelIm, stats, centroids = cv2.connectedComponentsWithStats(255-mask)

    #print(np.unique(labelIm))
    #print(np.sum(mask==0))
    # Avoid first centroid, unbounded component
    for j in range(1,len(np.unique(labelIm))):
        if stats[j][4] < areaTH or stats[j][2] < thicknessTH or stats[j][3] < thicknessTH:
            mask[labelIm == j] = 255
            #print("erasing "+str(j))
            #print(np.sum(mask==0))


def recoupMasks(masks, weights, th):
    """
    Function to combine a list of
    masks from different methods
    receives the masks and a list
    of weights and does a weighted
    addition of the masks.
    output a mask with the pixels over a threshold
    """
    def processPair(x):
        nonlocal ret
        m,w = x
        ret[m==0]+=w

    # initialize mask
    ret = masks[0].copy()
    ret[ret>0] = 0
    # process
    list(map(processPair,zip(masks,weights)))
    # now transform into binary mask
    ret[ret<th] = 0
    ret[ret>=th] = 255
    return 255 - ret

def strictBinarization(im):
    """
    Make sure an image is properly binarize
    works in-place
    """
    im[im<10]=0
    im[im>1]=255

def eraseSmallRegions(im,numPixels=2500):
    """
    Function to erase regions smaller than
    a given number of pixels
    """
    def processComponent(l):
        """
        Inner function to count
        pixels in component and
        erase small components
        """
        nonlocal im
        if np.count_nonzero(labelImage==l) < numPixels:
            im[labelImage==l]=255

    # Threshold  the image to make sure it is binary
    strictBinarization(im)
    #compute connected components
    numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(255-im)
    #traverse all labels but ignore label 0 as it contains the background
    list(map(processComponent,range(1,numLabels)))

def eraseNonFatRegions(im,fatness):
    """
    Function to erase regions not fat enough
    """
    def processComponent(l):
        """
        Inner function to count
        pixels in component and
        erase non-fat components
        """
        nonlocal im
        width=stats[l,cv2.CC_STAT_WIDTH]
        height=stats[l,cv2.CC_STAT_HEIGHT]
        area=stats[l,cv2.CC_STAT_AREA]
        if area<fatness*width*width or area<fatness*height*height :
            im[labelImage==l]=255

    # Threshold  the image to make sure it is binary
    strictBinarization(im)
    #compute connected components
    numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(255-im)
    list(map(processComponent,range(1,numLabels)))

def boxesFound(im1, im2, verbose = False):
    """
    Function that receives two bounding box
    images and counts how many of the boxes
    in the first image are also on the second
    """
    def processRegion(centroid):
        """
        inner function to count how
        many boxes in one image
        are also in the other (centroid)
        """
        nonlocal im2
        x,y = centroid
        # Not sure why the cast to int is needed (but it is)
        return (int)(im2[int(y), int(x)]) == 0
    # Threshold  the image to make sure it is binary
    strictBinarization(im1)
    strictBinarization(im2)

    numLabels, labelImage, stats, centroids = cv2.connectedComponentsWithStats(255-im1)
    totalBoxes = len(centroids)-1
    count = sum(list(map(processRegion,centroids[1:])))
    #print(count)
    #print(totalBoxes)
    if verbose: print("Result:"+str(100 * count/totalBoxes)+"%\n")

    if totalBoxes != 0:
        return 100 * count/totalBoxes
    else:
        return 0

def boxListEvaluation(bPred, bGT,th = 50):
    """
        receives two lists of boxes (predicted and ground truth)
        in x1,y1,x2,y2 format and outputs precision, recall,
    """
    def center(b):
        """
            returns the center of a box in x1,y1,x2,y2 format
        """
        return b[0]+(b[2]-b[0])/2,b[1]+(b[3]-b[1])/2
    def isTrueP(b,gtB):
        """
            goes over all boxes in the ground truth and checks
            if they overlap with the current box more than the threshold
        """
        for box in gtB:
            op = overlappingAreaPercentage(b,box)
            #print("overlap percentage "+str(op))
            if op>th: return True
        return False

    def overlappingAreaPercentage(b1, b2):
        """
        Compute the percentage of overlap of rect1 over rect2.

        Parameters:
            rect1: tuple (xmin, ymin, xmax, ymax) - First rectangle
            rect2: tuple (xmin, ymin, xmax, ymax) - Second rectangle

        Returns:
            float: Percentage of overlap (0-100) of rect1 over rect2.
        """
        # Extract coordinates
        x1_min, y1_min, x1_max, y1_max = b1
        x2_min, y2_min, x2_max, y2_max = b2

        # Compute the intersection rectangle
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        # Compute width and height of the intersection rectangle
        inter_width = max(0, inter_xmax - inter_xmin)
        inter_height = max(0, inter_ymax - inter_ymin)

        # Compute the area of intersection
        inter_area = inter_width * inter_height

        # Compute the area of the second rectangle (rect2)
        rect2_area = max(0, x2_max - x2_min) * max(0, y2_max - y2_min)

        # Avoid division by zero
        if rect2_area == 0:
            return 0.0

        # Calculate the overlap percentage
        overlap_percentage = (inter_area / rect2_area) * 100

        return overlap_percentage

    num_tp = 0
    for box in bPred:
        # decide if it is a TP or FP.
        isTP = isTrueP(box,bGT)
        if isTP: num_tp+=1

    print("found TP "+str(num_tp)+" of predictions "+str(len(bPred))+" and real objects "+str(len(bGT)))
    recall = num_tp/len(bGT)
    precision = num_tp/len(bPred)

    return precision,recall


if __name__ == '__main__':
    im = read_Binary_Mask(sys.argv[1])
    mask = read_Binary_Mask(sys.argv[2])
    folder = sys.argv[3]
    predictionsToKanjiImages(im, mask,folder,os.path.basename(sys.argv[1])[:-4],True)
