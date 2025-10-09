import numpy as np
import cv2
import sys
from math import sqrt
import os
from pathlib import Path
import re
import torch

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

#Read grayscale image
def read_grayscale(path):

    retVal = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if retVal is None: raise Exception("Reading GRAYSCALE image, something went wrong with the file name "+str(path))
    return retVal

#Read binary image
def read_Binary_Mask(path):

    retVal = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if retVal is None: raise Exception("Reading binary image, something went wrong with the file name "+str(path))

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

def boxListEvaluation(bPred, bGT,th = 0.5):
    """
        receives two lists of boxes (predicted and ground truth)
        in x1,y1,x2,y2 format and outputs number of TP, FP, FN
        in terms of overlap percentage
    """
    def isTrueP(b,gtB):
        """
            goes over all boxes in the ground truth and checks
            if they overlap with the current box more than the threshold
            store them in a dictionary
        """
        for boxGT in gtB:
            op = iou(b,boxGT)
            #print("overlap percentage "+str(op))
            if op>th and str(boxGT) not in tpDict:
                tpDict[str(boxGT)] = True


    def iou(b1, b2):
        """
        Compute IoU between two boxes.

        b1: list or tuple [x1_min, y1_min, x1_max, y1_max]
        b2: list or tuple [x2_min, y2_min, x2_max, y2_max]
        """
        x1_min, y1_min, x1_max, y1_max = b1
        x2_min, y2_min, x2_max, y2_max = b2

        # Compute intersection coordinates
        xi1 = max(x1_min, x2_min)
        yi1 = max(y1_min, y2_min)
        xi2 = min(x1_max, x2_max)
        yi2 = min(y1_max, y2_max)

        # Compute intersection area
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Compute union area
        b1_area = (x1_max - x1_min) * (y1_max - y1_min)
        b2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = b1_area + b2_area - inter_area

        # Avoid division by zero
        if union_area == 0:
            return 0.0

        return inter_area / union_area

    """
    print("predicted boxes \n")
    print(bPred)

    print("\nGT boxes!!!!\n")
    print(bGT)
    """
    # Dictionary that contains the boxes in the ground truth that have been overlapped by a predicted box
    tpDict = {}
    for box in bPred:
        isTrueP(box,bGT) # this function already updates the tpDict
    TP = len(tpDict.keys()) # we consider the number of true positives to be the number of ground truth boxes caught by predictions, any box caught more than once is only counted once
    FP = len(bPred) - TP # We consider a false positive to be either a box that misses the ground truth or repeats it
    FN = len(bGT) - TP # any missed box is a false negative

    #print(tpDict)
    #print("found TP "+str(num_tp)+" of predictions "+str(len(bPred))+" and real objects "+str(len(bGT)))
    #recall = num_tpRECALL/len(bGT) if len(bGT) > 0 else 0
    #precision = num_tp/len(bPred) if len(bPred) > 0 else 0
    #print("boxlistEval found "+str(TP)+" missed "+str(FN))
    return TP, FP, FN

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
    recall = num_tp/len(bGT) if len(bGT) > 0 else 0
    precision = num_tp/len(bPred) if len(bPred) > 0 else 0

    return precision,recall

def rebuildImageFromTiles(imageN, TileList, predFolder, origFolder):
    """
        Receive an imageName (image to build)
        and a list of tiles (as file Names)
        those tile names contain the first
        image name
    """
    def get_original_image_size(filenames, folder_path):
        """
        Compute size of original image based on tiles (OpenCV).

        Args:
            filenames (list of str): List of filenames like 'x0y0.jpg', 'x200y150.png', etc.
            folder_path (str): Path to the folder containing these image tiles.

        Returns:
            (width, height): Total width and height of the original image.
        """
        max_right = 0
        max_bottom = 0

        for fname in filenames:
            # Extract x and y using regex
            match = re.search(r'x(\d+)y(\d+)', fname)
            if not match:
                raise ValueError(f"Filename '{fname}' does not match 'x<num>y<num>' pattern.")

            x, y = int(match.group(1)), int(match.group(2))

            # Read image using OpenCV
            image_path = os.path.join(folder_path, fname)
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")

            tile_height, tile_width = img.shape[:2]

            # Compute max extent of the image
            max_right = max(max_right, x + tile_width)
            max_bottom = max(max_bottom, y + tile_height)

        return max_right, max_bottom

    #print("rebuild image")
    # Get final image dimensions
    full_width, full_height = get_original_image_size(TileList, origFolder)
    #print("rebuild image size "+str((full_width, full_height)))

    # Prepare white canvas
    #stitched_image = np.zeros((full_height, full_width, 3), dtype=np.uint8)
    stitched_image = np.full((full_height, full_width, 3), 255, dtype=np.uint8)
    stitched_mask = np.ones((full_height, full_width), dtype=np.uint8)
    stitched_mask = stitched_mask*255

    # list of boxes in the original image coordinates
    boxCoords = []

    for fname in TileList:
        #print("rebuild, processing image "+fname)
        match = re.search(r'x(\d+)y(\d+)', fname)
        if not match:
            raise ValueError(f"Filename '{fname}' does not contain valid 'x<num>y<num>' format.")

        x, y = int(match.group(1)), int(match.group(2))
        #image_path = os.path.join(predFolder, fname)
        image_path = os.path.join(origFolder, fname)
        #print("reading "+str(image_path))

        # Here the tiles should be read from the original folder, not the predicted one,
        tile = cv2.imread(image_path)
        if tile is None:
            # this should not happen
            raise FileNotFoundError(f"Could not read image: {image_path}")

        h, w = tile.shape[:2]
        stitched_image[y:y+h, x:x+w] = tile

        # the tile masks should be read from the predictions folder but they may not exist
        tileMask = cv2.imread(os.path.join(predFolder, "PREDMASK"+fname),0)

        #if tileMask is not None: print("tileMask read "+str(tileMask.shape) )

        # make sure to overlap all mask predictions
        stitched_maskAUX = np.ones((full_height, full_width), dtype=np.uint8)

        if tileMask is not None:
            h, w = min(h,tileMask.shape[0]) , min(2,tileMask.shape[1]) # not sure if this is really working, the tiles are pretty wonky
            stitched_maskAUX[y:y+h, x:x+w] = tileMask[:h,:w] #used to be just tilemask, check that this works
        stitched_mask[ stitched_maskAUX == 0 ] = 0


        # also read box coords if we have them
        if tileMask is not None:
            with open(os.path.join(predFolder, "BOXCOORDS"+fname[:-4]+".txt")) as f:
                for line in f.readlines():
                    c,px1,py1,px2,py2 = tuple(line.strip().split(" "))
                    newP1 = (int(float(px1) + float(x)), int(float(py1) + float(y)))
                    newP2 = (int(float(px2) + float(x)), int(float(py2) + float(y)))
                    boxCoords.append((c,str(newP1[0]),str(newP1[1]),str(newP2[0]),str(newP2[1])))

    #print("now going to write")

    # write to disk (image, mask, bounding box file)
    cv2.imwrite(os.path.join(predFolder,"FULL",imageN), stitched_image )
    cv2.imwrite(os.path.join(predFolder,"FULL","PREDMASK"+imageN), stitched_mask )

    #also Reshape the mask to predefined size
    reshapedMask =  cv2.resize(stitched_mask, (9922,7012), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(predFolder,"FULL","BIGMASK"+imageN), reshapedMask )

    boxCoordsToFile(os.path.join(predFolder,"FULL","BOXCOORDS"+imageN[:-4]+".txt"),boxCoords)
    # also, make a pretty image of the original image with boxes and categories
    cv2.imwrite(os.path.join(predFolder,"FULL","Pretty"+imageN), prettyImage(boxCoords,stitched_image) )
    #print("end rebuild image")


def prettyImage(boxes, image, color = 125, thickness=4, font_scale=0.5, font_thickness=3):
    """
        Draw bounding box countours on image
        box format is p1x,p1y,p2x,p2y
    """
    for tup in boxes:
        category = tup[0]
        px1, py1, px2, py2 = map(float, tup[1:])
        top_left = (int(px1), int(py1))
        bottom_right = (int(px2), int(py2))

        # Draw rectangle
        cv2.rectangle(image, top_left, bottom_right, color, thickness)

        # Put label
        label = str(category)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        label_origin = (top_left[0], top_left[1] - 5 if top_left[1] - 5 > 0 else top_left[1] + label_size[1] + 5)

        cv2.putText(image, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, font_thickness, lineType=cv2.LINE_AA)

    return image

def filter_boxes_by_overlap_and_area_distance(
    categories,
    boxes,
    image,  # Grayscale image, np.uint8 [H,W]
    output_file_name
):
    """
    Filters overlapping boxes by area distance to mean.
    Overlays removed boxes (gray) and kept boxes (black) on the grayscale image.
    Saves result as a single-channel PNG.
    """

    # Ensure boxes is tensor [N, 4]
    if isinstance(boxes, torch.Tensor):
        boxes_tensor = boxes
    else:
        boxes_tensor = torch.stack(boxes)

    num_boxes = boxes_tensor.shape[0]
    areas = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
    avg_area = areas.mean().item()

    keep = [True] * num_boxes
    involved = [False] * num_boxes

    for i in range(num_boxes):
        if not keep[i]:
            continue
        box_i = boxes_tensor[i]
        area_i = areas[i]
        for j in range(i + 1, num_boxes):
            if not keep[j]:
                continue
            box_j = boxes_tensor[j]
            area_j = areas[j]

            # intersection
            x1 = max(box_i[0], box_j[0])
            y1 = max(box_i[1], box_j[1])
            x2 = min(box_i[2], box_j[2])
            y2 = min(box_i[3], box_j[3])
            inter_w = max(0, x2 - x1)
            inter_h = max(0, y2 - y1)
            intersection = inter_w * inter_h

            overlap = intersection / min(area_i, area_j)

            if overlap > 0.5:
                involved[i] = True
                involved[j] = True
                if abs(area_i - avg_area) < abs(area_j - avg_area):
                    keep[j] = False
                else:
                    keep[i] = False

    kept_categories = [cat for k, cat in zip(keep, categories) if k]
    kept_boxes = [boxes_tensor[idx] for idx in range(num_boxes) if keep[idx]]

    # Make a copy of the input grayscale image
    overlay = image.copy()

    # Draw boxes
    for idx in range(num_boxes):
        if involved[idx]:
            x1, y1, x2, y2 = [int(v.item()) for v in boxes_tensor[idx]]
            if keep[idx]:
                # kept box -> black, thicker
                color = 0
                thickness = 3
            else:
                # removed box -> gray, thin
                color = 128
                thickness = 1
            # Draw rectangle in-place on the grayscale overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

    # Save single-channel PNG
    cv2.imwrite(output_file_name, overlay)

    return kept_categories, kept_boxes

def fillHolesInGrid(categories, boxes, image, output_mask_file="filled_mask.png", overlap_thresh=0.05):
    """
    Improved: robust overlay with type matching.
    """

    if isinstance(categories, list):
        categories = torch.cat(categories)
    if isinstance(boxes, list):
        boxes = torch.stack(boxes).float()
    else:
        boxes = boxes.float()

    if boxes.numel() == 0:
        print("No input boxes — saving blank mask and returning unchanged")
        mask = np.ones_like(image, dtype=np.uint8) * 255
        cv2.imwrite(output_mask_file, mask)
        cv2.imwrite(output_overlay_file, image)
        return categories, boxes

    # Compute robust average area
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_areas, _ = torch.sort(areas)
    n = len(areas)
    lo, hi = int(0.1 * n), int(0.9 * n)
    avArea = sorted_areas[lo:hi].mean().item()

    # Group by x to form columns
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2

    sorted_idx = torch.argsort(cx)
    cx_sorted = cx[sorted_idx]

    columns = []
    if len(cx_sorted) > 0:
        tol = 20
        col = [sorted_idx[0].item()]
        for i in range(1, len(cx_sorted)):
            if abs(cx_sorted[i] - cx_sorted[i - 1]) < tol:
                col.append(sorted_idx[i].item())
            else:
                columns.append(col)
                col = [sorted_idx[i].item()]
        columns.append(col)

    new_boxes = []
    new_cats = []

    side = np.sqrt(avArea)

    H, W = image.shape[:2]

    for col_idxs in columns:
        if len(col_idxs) < 2:
            continue

        col_cx = cx[col_idxs]
        col_cy = cy[col_idxs]

        sorty = torch.argsort(col_cy)
        col_cx = col_cx[sorty]
        col_cy = col_cy[sorty]

        for i in range(len(col_cy) - 1):
            top = col_cy[i].item()
            bot = col_cy[i + 1].item()
            gap = bot - top

            if gap > 1.2 * side:
                new_cy = top + side
                new_cx = col_cx[i].item()

                x1 = int(np.clip(new_cx - side / 2, 0, W))
                y1 = int(np.clip(new_cy - side / 2, 0, H))
                x2 = int(np.clip(new_cx + side / 2, 0, W))
                y2 = int(np.clip(new_cy + side / 2, 0, H))

                candidate = torch.tensor([x1, y1, x2, y2]).float()

                xx1 = torch.maximum(candidate[0], boxes[:, 0])
                yy1 = torch.maximum(candidate[1], boxes[:, 1])
                xx2 = torch.minimum(candidate[2], boxes[:, 2])
                yy2 = torch.minimum(candidate[3], boxes[:, 3])

                inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
                max_overlap = inter.max().item() / avArea

                if max_overlap > overlap_thresh:
                    continue

                new_boxes.append(candidate.unsqueeze(0))
                new_cats.append(torch.tensor([0]))  # dummy label

    if new_boxes:
        added_boxes = torch.cat(new_boxes, dim=0)
        added_cats = torch.cat(new_cats, dim=0)
        final_boxes = torch.cat([boxes, added_boxes], dim=0)
        final_cats = torch.cat([categories, added_cats], dim=0)
    else:
        added_boxes = torch.zeros((0, 4))
        final_boxes = boxes
        final_cats = categories

    mask = np.ones((H, W), dtype=np.uint8) * 255

    for box in boxes.int().cpu().numpy():
        x1, y1, x2, y2 = box
        cv2.rectangle(mask, (x1, y1), (x2, y2), 180, thickness=-1)

    if added_boxes.numel() > 0:
        for box in added_boxes.int().cpu().numpy():
            x1, y1, x2, y2 = box
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, thickness=-1)

    #cv2.imwrite(output_mask_file, mask)
    #print(f"Saved mask to: {output_mask_file}")

    # --- Robust overlay ---
    if image.ndim == 2:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = image.copy()

    if img_bgr.dtype != np.uint8:
        img_bgr = (np.clip(img_bgr, 0, 1) * 255).astype(np.uint8)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(img_bgr, 0.7, mask_bgr, 0.3, 0)
    cv2.imwrite(output_mask_file, overlay)
    #print(f"Saved overlay to: {output_overlay_file}")

    return final_cats, final_boxes


def maskFromBoxes(boxes, image_size):
    """
    Create a binary mask from bounding boxes.

    Args:
        boxes (list of lists or tensors): List of [x1, y1, x2, y2] boxes.
        image_size (tuple): (height, width) of the output mask.

    Returns:
        mask (np.ndarray): Binary mask with 1s inside boxes, 0 elsewhere.
    """
    height, width = image_size[0],image_size[1]
    mask = np.zeros((height, width), dtype=np.uint8)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        mask[y1:y2, x1:x2] = 255

    return 255-mask


def boxesFromMask(img, cl = 1, yoloFormat = True):
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

def borderbox(box, width, height, border_fraction=0.05):
    """
    Return True if a box touches the border region defined by border_fraction.
    """
    bx, by = border_fraction * width, border_fraction * height
    return (box[0] <= bx) or (box[2] >= width - bx) or (box[1] <= by) or (box[3] >= height - by)

def sliceAndBox(im,mask,slice):
    """
    Given image and mask, slice them
    output image slices and mask slice
    and text files
    """
    out = []
    wSize = (slice,slice)
    # make tiles with large overlap so we have fewer problems of double boxes
    for (x, y, window) in sliding_window(im, stepSize = int(slice*0.8), windowSize = wSize ):
        boxList = []
        # get mask window
        maskW = mask[y:y + wSize[1], x:x + wSize[0]]
        # The mask was already binarized
        # compute box coords in yolo format because only YOLO will read them
        coords = boxesFromMask(maskW, cl = 0, yoloFormat = True)
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

"""
def fileToBoxCoords(file, returnCat=False):
    Reads bounding boxes from a file.
    If returnCat=True: returns (c, px, py, w, h)
    If returnCat=False: returns (px, py, w, h)
    with open(file, 'r') as f:
        if returnCat:
            return [tuple(map(float, line.strip().split())) for line in f if line.strip()]
        else:
            return [tuple(map(float, line.strip().split()[1:])) for line in f if line.strip()]
"""

def fileToBoxCoords(file, returnCat=False, yoloToXYXY=False, imgSize=None):
    """
        Reads bounding boxes from a file.
        If returnCat=True: returns (c, px, py, w, h) else (px, py, w, h)
        the YOLOToXYXY thing changes from the annoying yolo format to something normal
        but it needs the image size
    """
    if yoloToXYXY and not imgSize:
        raise ValueError("imgSize=(w, h) required if yoloToXYXY=True")

    w_img, h_img = imgSize if imgSize else (1, 1)

    with open(file, 'r') as f:
        boxes = []
        for line in f:
            if not line.strip():
                continue
            vals = list(map(float, line.strip().split()))
            c, cx, cy, w, h = vals if returnCat else (None, *vals[1:])
            cx, cy, w, h = cx * w_img, cy * h_img, w * w_img, h * h_img if yoloToXYXY else (cx, cy, w, h)
            box = (cx - w/2, cy - h/2, cx + w/2, cy + h/2) if yoloToXYXY else (cx, cy, w, h)
            boxes.append((int(c), *box) if returnCat else box)
    return boxes



if __name__ == "__main__":
    #color = True
    #resampleTestFolder(sys.argv[1],float(sys.argv[2]),color)

    # single image clean up small regions
    #mask = read_Binary_Mask(sys.argv[1])
    #im = color_to_gray(read_Color_Image(sys.argv[2]))
    #cv2.imwrite(sys.argv[3],cleanUpMaskBlackPixels( mask , im , 100))

    # folder clean up black pixels, careful as it overwrites.
    cleanUpFolderBlackPixels(sys.argv[1], sakuma1 = True)
