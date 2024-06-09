"""
   File to do exmperiments in Kanji
   Detection methods
   In Wasan documents
"""

import configparser
import sys
import os
import cv2
from pathlib import Path
from itertools import product

from main import read_config
from predict import detectBlobsMSER,detectBlobsDOG
from imageUtils import boxesFound,read_Binary_Mask

def makeParamDicts(pars,vals):
    """
        Receives a list with parameter names
        and a list of list with values
        for each parameter

        Creates a list of dictionaries
        with the combinations of parameters
    """
    prod = list(product(*vals))
    res = [dict(zip(pars,tup)) for tup in prod]
    return res

def main(fName):
    """
    main function, read command line parameters
    and call the adequate functions
    """
    conf = read_config(fName)
    print(conf)

    imageFolder = conf["Train_input_dir_images"]
    maskFolder = conf["Train_input_dir_masks"]
    expFolder = conf["Exp_dir"]

    # make experiments directory if it did not exist
    Path(expFolder).mkdir(parents=True, exist_ok=True)

    # params
    params = ["over","min_s","max_s"]
    vals = [[0.1,0.3,0.5],[10,20,30,40],[40,50,100]]
    parDicts = makeParamDicts(params,vals)
    #https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_log

    paramsMSER = ["delta","minA","maxA"]
    valsMSER = [[5,15,20,50],[100,500,1000,2000,5000],[10000,14400,20000,25000]]
    #valsMSER = [[5,7],[60],[14400]]
    parDictsMSER = makeParamDicts(paramsMSER,valsMSER)
    #https://docs.opencv.org/3.4/d3/d28/classcv_1_1MSER.html
    #https://www.vlfeat.org/overview/mser.html
    #https://stackoverflow.com/questions/17647500/exact-meaning-of-the-parameters-given-to-initialize-mser-in-opencv-2-4-x

    testPref = "pastanaga"
    outFileDIRDOG = os.path.join(expFolder,testPref+"DOGDIR.txt")
    outFileINVDOG = os.path.join(expFolder,testPref+"DOGINV.txt")
    fDirDOG = open(outFileDIRDOG,'w+')
    fInvDOG = open(outFileINVDOG,'w+')

    outFileDIRMSER = os.path.join(expFolder,testPref+"MSERDIR.txt")
    outFileINVMSER = os.path.join(expFolder,testPref+"MSERINV.txt")
    fDirMSER = open(outFileDIRMSER,'w+')
    fInvMSER = open(outFileINVMSER,'w+')

    fDirDOG.write("IMAGE,"+str(parDicts)+"\n")
    fInvDOG.write("IMAGE,"+str(parDicts)+"\n")
    fDirMSER.write("IMAGE,"+str(parDictsMSER)+"\n")
    fInvMSER.write("IMAGE,"+str(parDictsMSER)+"\n")

    # traverse image folder
    for dirpath, dnames, fnames in os.walk(imageFolder):
        for f in fnames:
            print("processing "+f)
            # for each image, check if it has been noise removed
            # and if it has a mask
            if "noiseRemoval" in str(f):
                maskName = "KP"+str(f[:-30])+"AN.jpg"
                if Path(os.path.join(maskFolder,maskName)).is_file():
                    print("doing this one")
                    # we now have a function with ground truth
                    im = read_Binary_Mask(os.path.join(imageFolder,f))
                    mask = read_Binary_Mask(os.path.join(maskFolder,maskName))

                    # Masks are not the perfect size, reshape
                    mask = cv2.resize(mask, (im.shape[1],im.shape[0]))

                    fDirDOG.write(maskName)
                    fInvDOG.write(maskName)
                    fDirMSER.write(maskName)
                    fInvMSER.write(maskName)

                    for d in parDictsMSER:
                        print("Detecting blobs with MSER "+str(d))
                        outMask = detectBlobsMSER(im,d)
                        fDirMSER.write(","+f'{float(boxesFound(mask,outMask)):.2f}')
                        fInvMSER.write(","+f'{float(boxesFound(outMask,mask)):.2f}')

                    for d in parDicts:
                        print("Detecting blobs with "+str(d))
                        outMask = detectBlobsDOG(im,d)
                        fDirDOG.write(","+f'{float(boxesFound(mask,outMask)):.2f}')
                        fInvDOG.write(","+f'{float(boxesFound(outMask,mask)):.2f}')

                    # finish the line and flush
                    fDirMSER.write("\n")
                    fInvMSER.write("\n")
                    fDirMSER.flush()
                    fInvMSER.flush()

                    fDirDOG.write("\n")
                    fInvDOG.write("\n")
                    fDirDOG.flush()
                    fInvDOG.flush()
    #close output files
    fDirDOG.close()
    fInvDOG.close()
    fDirMSER.close()
    fInvMSER.close()

if __name__ == "__main__":

    # Configuration file name, can be entered in the command line
    configFile = "config.ini" if len(sys.argv) < 2 else sys.argv[1]
    main(configFile)



    #BEST
    #DOG 77.5665178571429	 {'over': 0.5;min_s': 20;max_s': 100}

    # MSER 72.6126785714286	 {'delta': 5;minA': 500;maxA': 25000}

