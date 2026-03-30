"""
   File to do experiments in Kanji
   Detection methods
   In Wasan documents
"""

import configparser
import sys
import os
import time
import cv2
import torch

from pathlib import Path
#from itertools import product

from datasets import ODDataset,ODDETRDataset

from config import read_config
from predict import detectBlobsMSER,detectBlobsDOG
from imageUtils import boxesFound,read_Binary_Mask,recoupMasks, boxesFromMask, boxListEvaluation, boxListEvaluationCentroids
from train import train_YOLO,makeTrainYAML, get_transform, train_pytorchModel,train_DETR

from dataHandlding import (buildTRVT,buildNewDataTesting,separateTrainTest, 
                           forPytorchFromYOLO, buildTestingFromSingleFolderSakuma2, 
                           buildTestingFromSingleFolderSakuma2NOGT,
                           makeParamDicts, paramsDictToString)
from predict import predict_yolo, predict_pytorch, predict_DETR

from transformers import DetrForObjectDetection, DetrImageProcessor
from experimentsUtils import MODULARDLExperiment
from itertools import product


def computeAndCombineMasks(file):
    """
    Receive a folder, compute all the masks for
    all files for two methods, MSER and DOG
    """
    conf = read_config(file)
    print(conf)

    imageFolder = conf["Train_input_dir_images"]
    maskFolder = conf["Train_input_dir_masks"]
    outFolder = conf["Masks_dir"]

    # make experiments directory if it did not exist
    Path(outFolder).mkdir(parents=True, exist_ok=True)

    # params
    parDictDOG = {"over":0.5,"min_s":20,"max_s":100}
    parDictMSER = {'delta': 5, "minA": 500,"maxA": 25000}

    # traverse image folder
    for dirpath, dnames, fnames in os.walk(imageFolder):
        for f in fnames:
            print("processing "+f)
            # for each image, check if it has been noise removed
            # and if it has a mask
            if "noiseRemoval" in str(f):
                maskName = "KP"+str(f[:-30])+"AN.jpg"
                if not Path(os.path.join(maskFolder,maskName)).is_file():
                    # do only images that have no original annotation mask
                    print("doing this one")
                    # we now have a function with ground truth
                    im = read_Binary_Mask(os.path.join(imageFolder,f))

                    # detec MSER
                    outMaskMSER = detectBlobsMSER(im,parDictMSER)
                    #cv2.imwrite(os.path.join(outFolder,"MASKMSER"+f+".png"), outMaskMSER )

                    # detectDOG
                    outMaskDOG = detectBlobsDOG(im,parDictDOG)
                    #cv2.imwrite(os.path.join(outFolder,"MASKDOG"+f+".png"),outMaskDOG )

                    # yolo masks are precomputed
                    yoloMask = cv2.imread(os.path.join(outFolder,"MASKcombined_data_200ex_"+str(f)+".png"),0)

                    recoupedMask = recoupMasks([outMaskDOG,outMaskMSER,yoloMask],[1,1,2],2)
                    cv2.imwrite(os.path.join(outFolder,"RECOUP"+f+".png"),recoupedMask )


def classicalDescriptorExperimentSakuma2(fName):
    """
    main function, read command line parameters
    and call the adequate functions
    """
    conf = read_config(fName)
    print(conf)

    imageFolder = conf["Test_input_dir"] # test input dir points to the images, 
    expFolder = conf["Exp_dir"]

    denoised = "denoised" in imageFolder

    # make experiments directory if it did not exist
    Path(expFolder).mkdir(parents=True, exist_ok=True)

    # params
    params = ["over","min_s","max_s"]
    vals = [[0.5],[30],[50]]
    #vals = [[0.1,0.3,0.5],[10,20,30,40],[40,50,100]]
    parDicts = makeParamDicts(params,vals)
    #https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_log

    paramsMSER = ["delta","minA","maxA"]
    valsMSER = [[20],[2000],[20000]]
    #valsMSER = [[5,15,20,50],[100,500,1000,2000,5000],[10000,14400,20000,25000]]

    parDictsMSER = makeParamDicts(paramsMSER,valsMSER)
    #https://docs.opencv.org/3.4/d3/d28/classcv_1_1MSER.html
    #https://www.vlfeat.org/overview/mser.html
    #https://stackoverflow.com/questions/17647500/exact-meaning-of-the-parameters-given-to-initialize-mser-in-opencv-2-4-x

    retMser = {}
    retDog = {}

    # traverse image folder
    for dirpath, dnames, fnames in os.walk(imageFolder):
        for f in fnames:
            #print("processing "+f)
            # identify all images that have masks (images are mixed, some have masks some don't)

            # for each image, check if it has been noise removed
            # and if it has a mask
            if "KP" in str(f):
                maskName = f
                imageName = maskName[2:-6]+"denoised.png" if denoised else maskName[2:-6]+".png"
                #print(maskName)
                #print(imageName)
                #sys.exit()
                #print("doing this one")
                # we now have a function with ground truth
                im = read_Binary_Mask(os.path.join(imageFolder,imageName))
                mask = read_Binary_Mask(os.path.join(imageFolder,maskName))

                # Masks are not the perfect size, reshape
                mask = cv2.resize(mask, (im.shape[1],im.shape[0]))

                gtBoxes = [t[1:] for t in  boxesFromMask(mask, cl = 1, yoloFormat = False)]# ignore category

                for d in parDictsMSER:
                    if str(d) not in retMser: retMser[str(d)] = [] 
                    print("Detecting blobs with MSER "+str(d))
                    outMask = detectBlobsMSER(im,d)

                    predBoxes = [t[1:] for t in  boxesFromMask(outMask, cl = 1, yoloFormat = False)]# ignore category
                    TP,FP,FN = boxListEvaluation(gtBoxes,predBoxes)
                    prec = TP/(TP+FP)
                    rec = TP/(TP+FN)
                    retMser[str(d)].append((prec,rec))
                    
                for d in parDicts:
                    print("Detecting blobs with "+str(d))
                    if str(d) not in retDog: retDog[str(d)] = [] 
                    outMask = detectBlobsDOG(im,d)
                    predBoxes = [t[1:] for t in  boxesFromMask(outMask, cl = 1, yoloFormat = False)]# ignore category
                  
                    TP,FP,FN = boxListEvaluation(gtBoxes,predBoxes)
                    prec = TP/(TP+FP)
                    rec = TP/(TP+FN)
                    retDog[str(d)].append((prec,rec))
 
    #print(retMser)
    #print(retDog)

    # Traverse all dictionary values
    for k,v in retMser.items():
       # Divide into sublists
       precList = [x[0] for x in v]
       recList = [x[1] for x in v]
       print(precList)
       print(recList)
       print("for MSER parameter values "+str(k)+" average precision was "+str(sum(precList)/len(precList))+" and the average recall was "+str(sum(recList)/len(recList)))     
    
    # Traverse all dictionary values
    for k,v in retDog.items():
       # Divide into sublists
       precList = [x[0] for x in v]
       recList = [x[1] for x in v]
       print(precList)
       print(recList)
       print("for DOG parameter values "+str(d)+" average precision was "+str(sum(precList)/len(precList))+" and the average recall was "+str(sum(recList)/len(recList)))     



def classicalDescriptorExperiment(fName):
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

    

# Single experiment:
if __name__ == "__main__":
    # Configuration file name, can be entered in the command line
    configFile = "config.ini" if len(sys.argv) < 2 else sys.argv[1]

    # classical descriptor experiment
    # classicalDescriptorExperimentSakuma2(configFile)

    print("IS CUDA AVAILABLE???????????????????????")
    print(torch.cuda.is_available())

    doYolo = False
    doPytorch = True
    doDETR = False

    # Configuration file name, can be entered in the command line
    configFile = "config.ini" if len(sys.argv) < 2 else sys.argv[1]

    # DL experiment
    conf = read_config(configFile)
    print(conf)
    
    # Define single parameter sets
    #yolo_params = {"scale": 0.45, "mosaic": 0} if doYolo else None
    #detr_params = {"modelType": "DETR", "lr": 5e-6, "batch_size": 8, "predconf": 0.5,"nms_iou": 0.5, "max_detections": 50} if doDETR else None

    # Run experiments
    for pars in product(["fcos","retinanet"],[0.005, 0.01], [50], [0.1, 0.5], [0.05, 0.1], [0.2, 0.3], [0.5, 0.7]):
    #for pars in product(["maskrcnn","fasterrcnn","ssd","fcos","retinanet","convnextmaskrcnn"],[0.005, 0.01, 0.001], [50, 100], [0.1, 0.5], [0.05, 0.1], [0.2, 0.3], [0.5, 0.7]):
    #for pars in product(["convnextmaskrcnn"],[0.005], [100], [0.1], [0.1], [0.3], [0.7]):
        mT, lr, step, gamma, score, nms, predconf = pars
        print(" @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ now experimenting with @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ "+str(pars))
        pytorch_params = {"modelType": mT, "score": score, "nms": nms, "predconf": predconf, "LR": lr, "STEP": step, "GAMMA": gamma}
        MODULARDLExperiment(conf, None, pytorch_params, None)
        print(" @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ END @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ "+str(pars))


#initial experiment
#if __name__ == "__main__":
#    print("IS CUDA AVAILABLE???????????????????????/")
#    print(torch.cuda.is_available())

    # Configuration file name, can be entered in the command line
#    configFile = "config.ini" if len(sys.argv) < 2 else sys.argv[1]

    #computeAndCombineMasks(configFile)

    # DL experiment
#    conf = read_config(configFile)
#    print(conf)

#    DLExperiment(conf, doYolo = False , doPytorchModels = False, doDETR = True)
#    DLExperiment(conf, doYolo = False , doPytorchModels = True, doDETR = False)
#    DLExperiment(conf, doYolo = True , doPytorchModels = False, doDETR = False)



#classicalDescriptorExperiment(configFile)
#BEST
#DOG 77.5665178571429	 {'over': 0.5;min_s': 20;max_s': 100}
# MSER 72.6126785714286	 {'delta': 5;minA': 500;maxA': 25000}
