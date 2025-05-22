import cv2
from skimage.feature import blob_dog
from math import sqrt
from pathlib import Path

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image,visualize_object_predictions
from sahi.predict import get_prediction,get_sliced_prediction,predict
import json
import numpy as np
import os
import torch
from train import collate_fn
from imageUtils import boxListEvaluation, boxListEvaluationCentroids, boxesFound, precRecall, maskFromBoxes,rebuildImageFromTiles

from torchvision.transforms.functional import to_pil_image

import sys
import time
from PIL import Image

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

# opencv MSER detector
def detectBlobsMSER(img, params_dict):
    #Adjust parameters
    if "delta" in params_dict: delt = params_dict["delta"]
    else:  delt = 5
    if "minA" in params_dict: minA = params_dict["minA"]
    else:  minA = 60
    if "maxA" in params_dict: maxA = params_dict["maxA"]
    else:  maxA = 14400
    # use MSER blob detector
    mser = cv2.MSER_create(delta = delt, min_area = minA, max_area = maxA)

    #detect regions in gray scale image
    regions, _ = mser.detectRegions(img)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    mask=255-mask

    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (0, 0, 0), -1)

    return mask

# SCIKITLEARN BLOB DETECTORS
def detectBlobsDOG(im, params_dict):

    if "min_s" in params_dict: min_s = params_dict["min_s"]
    else:  min_s=10
    if "max_s" in params_dict: max_s = params_dict["max_s"]
    else:  max_s=200
    if "over" in params_dict: over = params_dict["min_s"]
    else:  over=.1
    if "threshold" in params_dict: th = params_dict["min_s"]
    else:  th=.1


    blob_List = blob_dog(255-im, min_sigma = min_s, max_sigma = max_s, threshold=th, overlap = over)
    # Compute radii in the 3rd column.
    blob_List[:, 2] = blob_List[:, 2] * sqrt(2)

    mask = np.zeros((im.shape[0], im.shape[1], 1), dtype=np.uint8)
    # write to file
    for blob in blob_List:
        #print("blob!")
        y, x, r = blob
        cv2.rectangle(mask,(int(x-r), int(y-r)), (int(x+r), int(y+r)), 255, -1)

    return 255 - mask


# YOLO RELATED FUNCTIONS
def boxesToTextFile(result,file):
    """
    Receive a sahi prediction object and write
    the context to a legible text file
    """
    def writeBox(x):
        """
        inner function to write
        the information of a box
        to file
        """
        f.write(str(x.category.id)+" "+str(x.bbox.to_xywh()[0])+" "
        +str(x.bbox.to_xywh()[1])+" "+str(x.bbox.to_xywh()[2])+" "
        +str(x.bbox.to_xywh()[3])+"\n")

    with open(file,'w+') as f:
        list(map(writeBox,result.object_prediction_list))

def boxesToMaskFile(result,file,shape):
    """
    Receive a sahi prediction object and write
    the context to a Binary image
    """
    def paintBox(x):
        """
        Inner function to paint the info of
        a box into a binary mask
        """
        nonlocal im
        x,y,w,h =  x.bbox.to_xywh()
        im = cv2.rectangle(im, (int(x), int(y)),
        (int(x+w), int(y+h)), 0, -1)

    im = 255*np.ones((shape[0],shape[1]), np.uint8)
    with open(file,'w+') as f:
        list(map(paintBox,result.object_prediction_list))
        cv2.imwrite(file,im)
    return im

def boxesToPointsFile(result,file,shape):
    """
    Receive a sahi prediction object and write
    the context to a Binary
    """
    def paintCircle(x):
        """
        Inner function to paint the info of
        a box into a circle
        """
        nonlocal im
        x,y,w,h =  x.bbox.to_xywh()
        cv2.circle(im, (int(x), int(y)), 5, 0, -1)

    im = 255*np.ones((shape[0],shape[1]), np.uint8)
    with open(file,'w+') as f:
        list(map(paintCircle,result.object_prediction_list))
        cv2.imwrite(file,im)


def testFileList(folder):
    #print(folder)
    for dirpath, dnames, fnames in os.walk(folder):
        return [os.path.join(folder,f) for f in fnames]

def predict_new_Set_yolo(conf):
    """"
    receive a folder with images with no masks and predict them all
    """

    testPath = conf["Test_ND_dir"]
    testImageList = testFileList(testPath)
    modellist = conf["models"]
    predict_dir = conf["Pred_dir"]

    #print(testPath)
    #create predictions dir if it does not exist
    Path(predict_dir).mkdir(parents=True, exist_ok=True)
    dScore = []
    invScore = []
    ignoreCount = 0
    for imPath in testImageList:
        print("predicting "+imPath)
        for currentmodel in modellist: #not doing anything at the moment
            modelpath = conf["Train_res"]+"/detect/"+currentmodel+"/weights/best.pt"

            detectionModel = AutoDetectionModel.from_pretrained(model_type='yolov8',
                                                        model_path=modelpath,device=0)
            image = cv2.imread(imPath)

            result = get_sliced_prediction(image,detectionModel,slice_height=512
            ,slice_width=512,overlap_height_ratio=0.2,overlap_width_ratio=0.2,
            verbose = False )

            #boxesToTextFile(result,predict_dir+'/predictions_list_' + currentmodel + '_' + os.path.basename(imPath) +'.txt')
            outMask = boxesToMaskFile(result,predict_dir+'/PROVM' + currentmodel + '_' + os.path.basename(imPath) +'.png',image.shape)

def predict_yolo(conf, prefix = 'combined_data_'):

    testPath = os.path.join(conf["TV_dir"],conf["Test_dir"],"images")
    maskPath =os.path.join(conf["TV_dir"],conf["Test_dir"],"masks")
    testImageList = testFileList(testPath)
    modellist = conf["models"]
    predict_dir = conf["Pred_dir"]

    #print(testPath)
    #create predictions dir if it does not exist
    Path(predict_dir).mkdir(parents=True, exist_ok=True)
    dScore = []
    invScore = []
    ignoreCount = 0
    for imPath in testImageList:
        print("predictYOLO, predicting "+imPath)
        currentmodel = prefix if len(conf["models"])<1 else conf["models"][0] # should get totally rid of conf["models"]

        modelpath = conf["Train_res"]+"/detect/"+currentmodel+"/weights/best.pt"
        #print("predictYOLO, model path "+str(modelpath))

        detectionModel = AutoDetectionModel.from_pretrained(model_type='yolov8',
                                                    model_path=modelpath,device=0)
        image = cv2.imread(imPath)
        maskName = os.path.basename(imPath)[:-4]+"MASK.png"

        result = get_sliced_prediction(image,detectionModel,slice_height=512
        ,slice_width=512,overlap_height_ratio=0.2,overlap_width_ratio=0.2,
        verbose = False )

        boxesToTextFile(result,predict_dir+'/predictions_list_' + currentmodel + '_' + os.path.basename(imPath) +'.txt')
        outMask = boxesToMaskFile(result,predict_dir+'/MASK' + currentmodel + '_' + os.path.basename(imPath) +'.png',image.shape)

        # evaluate
        gtMaskPath = os.path.join(maskPath,maskName)
        gtMask = cv2.imread(gtMaskPath,0)
        try:
            dScore.append(boxesFound(gtMask,outMask, percentage = False))
            invScore.append(boxesFound(outMask,gtMask, percentage = False))
        except:
            print("image with no boxes, ignoring "+str(ignoreCount))
            ignoreCount+=1

        visualize_object_predictions(
            image=np.ascontiguousarray(result.image),
            object_prediction_list=result.object_prediction_list,
            rect_th=2,
            text_size=0.8,
            text_th=1,
            color=(255,0,0),
            output_dir=predict_dir,
            file_name='boxes_' + currentmodel + '_' +os.path.basename(imPath),
            export_format='png'
        )

    # computations
    #print(invScore)
    #print(dScore)
    prec, rec = precRecall(dScore, invScore)
    print("At the end of the test precision and recall values where "+str(prec)+" and "+str(rec))
    return prec,rec


@torch.no_grad()
def predict_pytorch_maskRCNN(dataset_test, model, device, predConfidence):
    """
        CURRENTLY NOT IN USE AS WE ARE FOCUSING IN DETECTION
        Inference for pytorch object detectors
        Currently not in use in favor of a simpler function
        This one accesses the masks in case we want to do real
        mask prediction and not only boxes
    """
    data_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    print("Testing Dataset Length "+str(len(dataset_test)))

    # evaluate on the test dataset
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    print("evaluating "+str(len(data_loader)))

    # store all images in disk (debugging purposes)
    #count = 0
    #for images,targets in data_loader:
    #    for im in images:
    #        to_pil_image(im).save("./debug/testIM"+str(count)+".png")
    #    for t in targets:

    #        mT = t["masks"]
    #        mT = (mT > 0.5).squeeze(1).cpu().numpy()
    #        aMT = mT[0]*255

            # store all ground truth masks too
            #for i, mT in enumerate(mT):
            #    aMT[mT>0]=255
            #gtmA = (255-aMT).astype("uint8")
            #gtm = Image.fromarray(gtmA, mode="L")

            #gtm.save("./debug/TestIM"+str(count)+"GTmask.png")

    #    count+=1

    count=0
    precList = []
    recList = []
    dScore = []
    invScore = []
    ignoreCount = 0

    for images, targets in data_loader:
        # store test images to disk
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # Move outputs to CPU and filter based on confidence score
        filtered_outputs = []
        for output in outputs:
            # Filter masks where confidence (score) is > 0.9
            high_conf_indices = output['scores'] > predConfidence

            # Keep only high-confidence masks, boxes, and labels
            filtered_output = {
                'boxes': output['boxes'][high_conf_indices].to(cpu_device),
                'labels': output['labels'][high_conf_indices].to(cpu_device),
                'scores': output['scores'][high_conf_indices].to(cpu_device),
                'masks': output['masks'][high_conf_indices].to(cpu_device),
            }
            filtered_outputs.append(filtered_output)

        model_time = time.time() - model_time

        evaluator_time = time.time()
        if len(filtered_outputs)>=1 and len(filtered_outputs[0]["masks"])>=1:
            masks = filtered_outputs[0]["masks"]

            # Threshold and convert to NumPy
            masks = (masks > 0.5).squeeze(1).cpu().numpy()
            accumMask = masks[0]*255
            for i, mask in enumerate(masks):
                accumMask[mask>0]=255

            # we need both pillow and array format
            outMaskArray = (255-accumMask).astype("uint8")
            outMask = Image.fromarray(outMaskArray, mode="L")

            outMask.save("./debug/TestIM"+str(count)+"mask.png")

            prec,rec = boxListEvaluation(outputs[0]["boxes"],targets[0]["boxes"])
            # create an output mask here and evaluate with boxesFound like for Pytorch
            masksT = targets[0]["masks"]
            masksT = (masksT > 0.5).squeeze(1).cpu().numpy()
            accumMaskT = masksT[0]*255
            for i, maskT in enumerate(masksT):
                accumMaskT[maskT>0]=255
            gtMaskArray = (255-accumMaskT).astype("uint8")
            gtMask = Image.fromarray(gtMaskArray, mode="L")

            # saving masks for debugging purposes
            gtMask.save("./debug/TestIM"+str(count)+"GTmask.png")

            try:
                dScore.append(boxesFound(gtMaskArray,outMaskArray, percentage = False))
                invScore.append(boxesFound(outMaskArray,gtMaskArray, percentage = False))
            except Exception as X:
                print(X)
                #print("image with no boxes, ignoring "+str(ignoreCount))
                ignoreCount+=1


            # also, write down the boxes in a text file
            sliceInfoDict = dataset_test.getSliceFileInfo()
            #print(sliceInfoDict)
        else:
            prec,rec = 0,0

        precList.append(prec)
        recList.append(rec)
        evaluator_time = time.time() - evaluator_time
        #print("time "+str(evaluator_time))
        count+=1


    # accumulate predictions from all images
    torch.set_num_threads(n_threads)

    # computations
    #print(invScore)
    #print(dScore)
    prec, rec = precRecall(dScore, invScore)
    print("At the end of the test precision and recall values (in terms of centroids) where "+str(prec)+" and "+str(rec))

    #print(precList)
    #print(recList)
    print("average Precision (overlap) "+str(sum(precList) / len(precList)))
    print("average Recall (overlap) "+str(sum(recList) / len(recList)))

    return prec,rec,sum(precList) / len(precList), sum(recList) / len(recList)

@torch.no_grad()
def predict_pytorch(dataset_test, model, device, predConfidence, predFolder):
    """
        Inference for pytorch object detectors
    """

    data_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    print("Testing Dataset Length "+str(len(dataset_test)))
    print("testing dataset image dict "+str(dataset_test.slicesToImages))

    # create output folder if necessary
    Path(os.path.join(predFolder,"FULL")).mkdir(parents=True, exist_ok=True)

    # evaluate on the test dataset
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    print("evaluating "+str(len(data_loader)))

    count=0
    precList = []
    recList = []
    dScore = []
    invScore = []
    ignoreCount = 0

    for ind, (images, targets) in enumerate(data_loader):
        imageName = dataset_test.imageNameList[ind].split(os.sep)[-1]
        imToStore = np.ascontiguousarray(images[0].permute(1, 2, 0))
       
        # store test images to disk
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # Move outputs to CPU and filter based on confidence score
        filtered_outputs = []
        for output in outputs:
            # Filter masks where confidence (score) is > 0.9
            high_conf_indices = output['scores'] > predConfidence

            # Keep only high-confidence masks, boxes, and labels
            filtered_output = {
                'boxes': output['boxes'][high_conf_indices].to(cpu_device),
                'labels': output['labels'][high_conf_indices].to(cpu_device),
                'scores': output['scores'][high_conf_indices].to(cpu_device),
            }
            filtered_outputs.append(filtered_output)


        model_time = time.time() - model_time

        evaluator_time = time.time()
        if len(filtered_outputs)>=1 :

            prec,rec = boxListEvaluation(filtered_outputs[0]["boxes"],targets[0]["boxes"])
            dS, invS = boxListEvaluationCentroids(filtered_outputs[0]["boxes"],targets[0]["boxes"])

            predMask = maskFromBoxes(filtered_outputs[0]["boxes"],imToStore.shape)

    
        else:
            prec,rec, dS, invS = 0,0,0,0
            predMask = maskFromBoxes([],imToStore.shape)

        precList.append(prec)
        recList.append(rec)
        dScore.append(dS)
        invScore.append(invS)
        # store image and predicted mask
        cv2.imwrite( os.path.join(predFolder,imageName), imToStore*255 )
        cv2.imwrite( os.path.join(predFolder,"PREDMASK"+imageName),predMask  )

        evaluator_time = time.time() - evaluator_time
        #print("time "+str(evaluator_time))
        count+=1

    # now reconstruct the full images and masks from what we have in the folder
    for imageN,TileList  in dataset_test.getSliceFileInfo().items():
        rebuildImageFromTiles(imageN,TileList,predFolder)



    # accumulate predictions from all images
    torch.set_num_threads(n_threads)

    # computations
    #print(invScore)
    #print(dScore)
    print("average Precision (centroids) "+str(sum(dScore) / len(dScore)))
    print("average Recall (centroids) "+str(sum(invScore) / len(invScore)))

    #print(precList)
    #print(recList)
    print("average Precision (overlap) "+str(sum(precList) / len(precList)))
    print("average Recall (overlap) "+str(sum(recList) / len(recList)))

    if sum(precList) / len(precList) >1: raise Exception("what is this shit?")

    return sum(dScore) / len(dScore), sum(invScore) / len(invScore) ,sum(precList) / len(precList), sum(recList) / len(recList)
