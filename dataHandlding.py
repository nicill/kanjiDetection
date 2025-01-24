"""
File to build training and Validation datasets
For Kanji Detection using deep learning
"""
from imageUtils import sliding_window,read_Color_Image,read_Binary_Mask,strictBinarization,sliceAndBox,boxCoordsToFile,boxesFromMask
import os
import cv2
import sys
from random import randint,uniform
import shutil
from pathlib import Path
from patch_classification import loadModelReadClassDict,testAndOutputForAnnotations

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


def foldersToAnnotationDB(maskFolder, imageFolder, outFolder, storeContext=False):
    """
        Traverse the annotation masks in maskFolder
        find the images with corresponding names in imageFolder
        for each image with mask, make a new folder with images 
        for every Kanji and context images if necessary
    """  
    # create output directory if it does not exist
    Path(outFolder).mkdir(parents=True, exist_ok=True)  

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

            # create a folder for the current image
            code = os.path.basename(imageName)[:-4]
            folder = os.path.join(outFolder,code)
            Path( folder ).mkdir(parents=True, exist_ok=True)

            # call the function to make the predictions if necessary
            predictionsToKanjiImages(im, mask, folder,code , storeContext)

def predictAllFolders(dataFolder, model,weights, classDict ):
    """
        Traverse a data folder containing 
        folders with all the separated Kanji for one image
        precit them and put the results in a text file
    """  

    for dirpath, dnames, fnames in os.walk(dataFolder):
        for d in dnames:
            print("Going to predict "+str(d))
            outFileName = d+".txt"

            testAndOutputForAnnotations(d,outFileName,model,weights, classDict)

 
def buildTrainValid(imageFolder, maskFolder, slice, outTrain, outVal, perc):
    """
        Receives a folder with images
        And another of masks
        builds a training and a validation set using all the
        files with masks
        CAREFUL! name correspondences between
        mask and image files
    """
    # create output directories if they do not exist
    for d in [os.path.join(outTrain,"images"),os.path.join(outTrain,"masks"),
    os.path.join(outTrain,"labels"),os.path.join(outVal,"images"),
    os.path.join(outVal,"masks"),os.path.join(outVal,"labels")]:
        Path(d).mkdir(parents=True, exist_ok=True)
    #for d in [os.path.join(outTrain,"images"),os.path.join(outTrain,"labels"),
    #os.path.join(outVal,"images"),os.path.join(outVal,"labels")]:
    #    Path(d).mkdir(parents=True, exist_ok=True)
    print("made paths "+str(os.path.join(outVal,"images") ))

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


def separateTrainTest(inFolder, outFolder, proportion = 0.9):
    """
    Given a Folder with Wasan images, separate a proportion of them into trainind and testing
    the folder has two subfolders, images and masks
    """
    # create output directories if they do not exist
    for d in [os.path.join(outFolder),os.path.join(outFolder,"train"), 
              os.path.join(outFolder,"train","images"),os.path.join(outFolder,"train","masks"),
              os.path.join(outFolder,"test"), os.path.join(outFolder,"test","images"),os.path.join(outFolder,"test","masks")]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("made "+str([os.path.join(outFolder),os.path.join(outFolder,"train"), os.path.join(outFolder,"train","images"),os.path.join(outFolder,"train","masks"),
              os.path.join(outFolder,"test"), os.path.join(outFolder,"test","images"),os.path.join(outFolder,"test","masks")]))    

    for dirpath, dnames, fnames in os.walk(os.path.join(inFolder,"masks")):
        for f in fnames:
            # f contains the mask name, make the image name
            imageName = f[2:-6]+".tif_resultat_noiseRemoval.tif"

            # random draw, test or train
            saveTo = os.path.join(outFolder,"train") if uniform(0, 1) > proportion else os.path.join(outFolder,"test")
            shutil.copyfile( os.path.join(inFolder,"masks",f), os.path.join(saveTo,"masks", f ))
            shutil.copyfile( os.path.join(inFolder,"images",imageName), os.path.join(saveTo,"images", imageName ))


if __name__ == '__main__':

    # To create the kanji folder of one image
    #im = read_Binary_Mask(sys.argv[1])
    #mask = read_Binary_Mask(sys.argv[2])
    #folder = sys.argv[3]
    #predictionsToKanjiImages(im, mask,folder,os.path.basename(sys.argv[1])[:-4],True)
    
    # to create the Kanji folder of the whole database
    maskFolder = sys.argv[1]
    imageFolder = sys.argv[2]
    outFolder = sys.argv[3]

    buildDB = False
    if buildDB:
        foldersToAnnotationDB(maskFolder, imageFolder, outFolder, storeContext=True)

    # loadmodelReadClassDict params: arch,modelFile,cDFile 
    mod,w,cD = loadModelReadClassDict(sys.argv[4], sys.argv[5], sys.argv[6])
    predictAllFolders(outFolder,mod,w,cD)
