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
import sys

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

def predict_yolo(conf):

    testPath = conf["Test_dir"]+"images"
    testImageList = testFileList(testPath)
    modellist = conf["models"]
    predict_dir = conf["Pred_dir"]

    #print(testPath)
    #create predictions dir if it does not exist
    Path(predict_dir).mkdir(parents=True, exist_ok=True)

    for imPath in testImageList:
        print("predicting "+imPath)
        for currentmodel in modellist: #not doing anything at the moment
            modelpath = conf["Train_res"]+"/detect/"+currentmodel+"/weights/best.pt"

            detectionModel = AutoDetectionModel.from_pretrained(model_type='yolov8',model_path=modelpath,device=0)
            image = cv2.imread(imPath)

            result = get_sliced_prediction(image,detectionModel,slice_height=512
            ,slice_width=512,overlap_height_ratio=0.2,overlap_width_ratio=0.2,
            verbose = False )

            boxesToTextFile(result,predict_dir+'/predictions_list_' + currentmodel + '_' + os.path.basename(imPath) +'.txt')
            boxesToMaskFile(result,predict_dir+'/MASK' + currentmodel + '_' + os.path.basename(imPath) +'.png',image.shape)

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

            #with open(predict_dir+'/predictions_list_' + currentmodel + '_' + os.path.basename(imPath) +'.json','w+') as resjson:
            #    s = json.dumps(result.to_coco_annotations())
            #    resjson.write(s)
