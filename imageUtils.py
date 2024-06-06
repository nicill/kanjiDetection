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
    """
    Euclidean Distance bewteen 2D points
    """
    return sqrt( (p[0]-q[0])*(p[0]-q[0])+(p[1]-q[1])*(p[1]-q[1]))

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
