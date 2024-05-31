import numpy as np
import cv2
import sys


from math import sqrt

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
    return sqrt( (p[0]-q[0])*(p[0]-q[0])+(p[1]-q[1])*(p[1]-q[1]))
