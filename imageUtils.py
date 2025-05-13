import numpy as np
import cv2
import sys
from math import sqrt
import os
from pathlib import Path

def resampleTestFolder(folder, factor, color = False):
    """
        Receive a folder related to a test (contains images, masks folders )
        Resamples all images according to the factor
        Puts the result in a new subfolder called "resampled"
    """
    outFolder = os.path.join(folder, "resampled")
    maskFolder = os.path.join(folder, "masks")
    imageFolder = os.path.join(folder, "images")

    outMaskFolder = os.path.join(outFolder, "masks")
    outImageFolder = os.path.join(outFolder, "images")

    for d in [outFolder,outMaskFolder,outImageFolder]:
        # create output directories if they do not exist
        Path(d).mkdir(parents=True, exist_ok=True)

    # traverse all original masks, resample them and their images
    for dirpath, dnames, fnames in os.walk(maskFolder):
        for f in fnames:
            # read mask and image, everyone is binary
            mask = read_Binary_Mask(os.path.join(maskFolder,f))
            imageName = f[:-8]+f[-4:] # images and masks must have the same extension
            im = read_Color_Image(os.path.join(imageFolder,imageName)) if color else read_Binary_Mask(os.path.join(imageFolder,imageName))

            # reshape with the factor
            mask = cv2.resize(mask, (int(im.shape[1]*factor),int(im.shape[0]*factor)))
            im = cv2.resize(im, (int(im.shape[1]*factor),int(im.shape[0]*factor)))

            #store the result
            cv2.imwrite(os.path.join(outImageFolder,imageName),im)
            cv2.imwrite(os.path.join(outMaskFolder,f),mask)


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

def color_to_gray(im):
    """
    receive a color image, turn it to grayscale
    """
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

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

def cleanUpMaskBlackPixels(mask, im, areaTH = 100):
    """
    Receive a Mask with the position of kanji
    and the original image (binarized)
    erase regions that have fewer black pixels 
    in the original image than a given threshold 
    """
    numLabels, labelIm, stats, centroids = cv2.connectedComponentsWithStats(255-mask)

    # Avoid first centroid, unbounded component
    for j in range(1,len(np.unique(labelIm))):
        #aux = im.copy() # copy image so we do not break anything

        blackInComponent = np.sum((labelIm == j ) & (im < 100 )) # count pixels in the mask that are black in the original image 
        if blackInComponent < areaTH:
            mask[labelIm == j] = 255
            #print("erasing "+str(j))
    return mask

def cleanUpFolderBlackPixels(folder, sakuma1 = False):
    """
        traverse a folder with particular naming conventions 
        ( annotations start with KP)
        and clean up the masks, OVERWRITES!
        avoids subfolders   
        the sakuma 1 flag is to process the older file naming
    """    
    for dirpath, dnames, fnames in os.walk(folder):
        for f in fnames:
            if "KP" in f: #annotation file
                print("fixing "+str(f))
                # read mask and image, everyone is binary
                mask = read_Binary_Mask(os.path.join(folder,f))
                imageName = f[2:-6]+".tif_resultat_noiseRemoval.tif" if sakuma1 else f[2:-6]+f[-4:] # images and masks must have the same extension
                im = read_Binary_Mask(os.path.join(folder,imageName)) if sakuma1 else color_to_gray(read_Color_Image(os.path.join(folder,imageName)) )
                cv2.imwrite(os.path.join(folder,f),cleanUpMaskBlackPixels( mask , im , 100)) 
        break # we do not want to chek subfolders


        
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
    images are opencv format, not pillow

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

def precRecall(dScore, invScore):
    """
        (Receive two list, the first with tuples of boxes found and total boxes 
        in the GT found in the prediction, dScore and viceversa, invScore in that order)
    """
    gtBoxes = sum([ x[1] for x in dScore ])
    foundGTBoxes = sum([ x[0] for x in dScore ])

    predBoxes = sum([ x[1] for x in invScore ])
    TPBoxes = sum([ x[0] for x in invScore ])

    #return precision and recall
    prec = 0 if predBoxes == 0 else 100*TPBoxes/predBoxes
    rec =  0 if gtBoxes == 0 else 100*foundGTBoxes/gtBoxes
    return prec , rec

def boxesFound(im1, im2, percentage = True, verbose = False):
    """
    Function that receives two bounding box
    images and counts how many of the boxes
    in the first image are also on the second
    """
    def processRegion(centroid):
        """
        inner function to count how
        many boxes in one image
        are also in the other 
        (regarding if the centroid is black)
        """
        nonlocal im2
        x,y = centroid
        return (int)(im2[int(y), int(x)]) == 0

    # Threshold  the image to make sure it is binary
    strictBinarization(im2)
    strictBinarization(im1)

    numLabels, labelImage, stats, centroids = cv2.connectedComponentsWithStats(255-im1)
    totalBoxes = len(centroids)-1
    count = sum(list(map(processRegion,centroids[1:])))
    #print(count)
    #print(totalBoxes)
    if verbose: print("Result:"+str(100 * count/totalBoxes)+"%\n")

    if percentage:
        if totalBoxes != 0:
            return 100 * count/totalBoxes
        else:
            raise Exception("Image with no boxes")
    else:
        return (count,totalBoxes)

def boxListEvaluation(bPred, bGT,th = 50):
    """
        receives two lists of boxes (predicted and ground truth)
        in x1,y1,x2,y2 format and outputs precision, recall,
        in terms of overlap percentage
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
            store them in a dictionary 
        """
        for boxGT in gtB:
            op = overlappingAreaPercentage(b,boxGT)
            #print("overlap percentage "+str(op))
            if op>th and str(boxGT) not in tpDict:
                    tpDict[str(boxGT)] = True
                

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


    # Dictionary that contains the boxes in the ground truth that have been overlapped by a predicted box
    tpDict = {}
    #num_tp = 0
    for box in bPred:
        # decide if it is a TP or FP.
        #isTP = isTrueP(box,bGT)
        #if isTP: num_tp+=1
        isTrueP(box,bGT)
    num_tp = len(tpDict.keys())    

    #print(tpDict)
    #print("found TP "+str(num_tp)+" of predictions "+str(len(bPred))+" and real objects "+str(len(bGT)))
    recall = num_tp/len(bGT)
    precision = num_tp/len(bPred)

    return precision,recall

def boxListEvaluationCentroids(bPred, bGT):
    """
        receives two lists of boxes (predicted and ground truth)
        in x1,y1,x2,y2 format and outputs precision, recall,
        in terms of centroids
    """
    def center(b):
        """
            returns the center of a box in x1,y1,x2,y2 format
        """
        return b[0]+(b[2]-b[0])/2,b[1]+(b[3]-b[1])/2
    
    def inside(b,p):
        """
        Check if p is inside box b
        """
        return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


    def isTrueP(b,gtB):
        """
            goes over all boxes in the ground truth and checks
            if any of them contains the centroid of the current box
        """
        c = center(b)
        for boxGT in gtB:
            if inside(boxGT,c) and str(boxGT) not in tpDict:
                    tpDict[str(boxGT)] = True
                
    # Dictionary that contains the boxes in the ground truth that have been overlapped by a predicted box
    tpDict = {}
    #num_tp = 0
    for box in bPred:
        # decide if it is a TP or FP.
        #isTP = isTrueP(box,bGT)
        #if isTP: num_tp+=1
        isTrueP(box,bGT)
    num_tp = len(tpDict.keys())    

    #print(tpDict)
    #print("found TP "+str(num_tp)+" of predictions "+str(len(bPred))+" and real objects "+str(len(bGT)))
    recall = num_tp/len(bGT)
    precision = num_tp/len(bPred)

    return precision,recall



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
        if stats[j][4] > threshold:
            if yoloFormat:
                h, w = img.shape
                bw = stats[j][2]
                bh = stats[j][3]
                cx = stats[j][0] + bw/2
                cy = stats[j][1] + bh/2
                # normalize and append
                out.append((cl,cx/w,cy/h,bw/w,bh/h))
            else:
                bw = stats[j][2]
                bh = stats[j][3]
                x1 = stats[j][0]
                y1 = stats[j][1]
                x2 = stats[j][0] + bw
                y2 = stats[j][1] + bh
                # append
                out.append((cl,x1,y1,x2,y2))
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

if __name__ == "__main__":
    #color = True
    #resampleTestFolder(sys.argv[1],float(sys.argv[2]),color)

    # single image clean up small regions
    #mask = read_Binary_Mask(sys.argv[1])
    #im = color_to_gray(read_Color_Image(sys.argv[2]))
    #cv2.imwrite(sys.argv[3],cleanUpMaskBlackPixels( mask , im , 100)) 

    # folder clean up black pixels, careful as it overwrites.
    cleanUpFolderBlackPixels(sys.argv[1], sakuma1 = True)