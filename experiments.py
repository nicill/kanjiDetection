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

from config import read_config
from predict import detectBlobsMSER,detectBlobsDOG
from imageUtils import boxesFound,read_Binary_Mask,recoupMasks,color_to_gray
from train import train_YOLO,makeTrainYAML

from dataHandlding import buildTRVT,buildNewDataTesting,separateTrainTest, forPytorchFromYOLO, buildTestingFromSingleFolderSakuma2
from predict import predict_yolo, predict_pytorch

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

def paramsDictToString(aDict, sep = ""):
    """
    Function to create a string from a params dict
    """
    ret = ""
    for k,v in aDict.items():
        ret+=str(k)+sep+str(v)+sep
    return ret[:-1]

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


def DLExperiment(conf, doYolo = False, doFRCNN = False):
    """
        Experiment to compare different values of DL networks
    """
    if  conf["Prep"]:
        # Compute train, validation, DO NOT do test
        buildTRVT(conf["Train_input_dir_images"], conf["Train_input_dir_masks"],conf["slice"],
        os.path.join(conf["TV_dir"],conf["Train_dir"]), os.path.join(conf["TV_dir"],conf["Valid_dir"]),
        os.path.join(conf["TV_dir"],conf["Test_dir"]),  conf["Train_Perc"], doTest = False)

        # Now test comes from another source
        # careful, this contains a hardcoded resampling factor!
        buildTestingFromSingleFolderSakuma2(conf["Test_input_dir"],os.path.join(conf["TV_dir"],conf["Test_dir"]),conf["slice"])

    f = open("outfile.txt","w+")
    if conf["Train"]:
        print("train YOLO ")
        # start YOLO experiment
        # Yolo Params is a list of dictionaries with all possible parameters
        yoloParams = makeParamDicts(["scale", "mosaic"],
                                    [[0.2,0.5,0.9],[0.0,0.5]])
        for params in yoloParams:
            # Train this version of the YOLO NETWORK
            yamlTrainFile = "trainEXP.yaml"
            prefix = "exp"+paramsDictToString(params)
            makeTrainYAML(conf,yamlTrainFile,params)
            train_YOLO(conf, yamlTrainFile, prefix)

            # Test this version of the YOLO Network
            print("TESTING YOLO!!!!!!!!!!!!!!!!!")
            prec,rec = predict_yolo(conf)
            f.write(str(paramsDictToString(params,sep=",")))

        sys.exit()
        print("train FRCNN")

        # our dataset has two classes only - background and Kanji
        num_classes = 2
        bs = 32 # should probably be a parameter
        proportion = conf["Train_Perc"]/100
        # parameter in case we want to separate or use the one created by YOLO

        dataset = ODDataset(os.path.join(conf["torchData"],"train"), yoloFormat, conf["slice"], get_transform())
        dataset_test = ODDataset(os.path.join(conf["torchData"],"test"), yoloFormat, conf["slice"], get_transform())
        # case for not yolo format, not in use I think
        #dataset = ODDataset(os.path.join(conf["torchData"],"separated","train"), yoloFormat, conf["slice"], get_transform())
        #dataset_test = ODDataset(os.path.join(conf["torchData"],"separated","test"), yoloFormat, conf["slice"], get_transform())


        frcnnParams = makeParamDicts(["score", "nms"],
                                    [[0.05,0.5],[0.25,0.5]])
        # score: Increase to filter out low-confidence boxes (default ~0.05)
        # nms: Reduce to suppress more overlapping boxes (default ~0.5)

        for tParams in frcnnParams:
            filePath = "exp"+paramsDictToString(tParams)+"fasterrcnn_resnet50_fpn.pth"
            #tParams = {"score":conf["pScoreTH"],"nms":conf["pnmsTH"]}
            # there is a proportion parameter that we may or may not want to touch
            pmodel = train_pytorchModel(dataset = dataset, device = device,
            num_classes = num_classes, file_path = conf["pmodel"],
            num_epochs = conf["ep"], trainAgain=conf["again"],
            proportion = proportion, trainParams = tParams)

    f.close()


if __name__ == "__main__":

    # Configuration file name, can be entered in the command line
    configFile = "config.ini" if len(sys.argv) < 2 else sys.argv[1]

    #computeAndCombineMasks(configFile)
    #classicalDescriptorExperiment(configFile)
    #BEST
    #DOG 77.5665178571429	 {'over': 0.5;min_s': 20;max_s': 100}
    # MSER 72.6126785714286	 {'delta': 5;minA': 500;maxA': 25000}


    # DL experiment
    conf = read_config(configFile)
    print(conf)

    DLExperiment(conf)
