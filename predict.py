import cv2
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image,visualize_object_predictions
from sahi.predict import get_prediction,get_sliced_prediction,predict
import json
import numpy as np
import os
import sys

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

def testFileList(folder):
    print(folder)
    for dirpath, dnames, fnames in os.walk(folder):
        return [os.path.join(folder,f) for f in fnames]

def predict_yolo(conf):

    testPath = conf["Test_dir"]+"images"
    testImageList = testFileList(testPath)
    modellist = conf["models"]
    predict_dir = conf["Pred_dir"]

    print(testPath)

    for imPath in testImageList:
        for currentmodel in modellist: #not doing anything at the moment
            modelpath = conf["Train_res"]+"/detect/"+currentmodel+"/weights/best.pt"

            print(modelpath)

            detectionModel = AutoDetectionModel.from_pretrained(model_type='yolov8',model_path=modelpath,device=0)
            image = cv2.imread(imPath)

            result = get_sliced_prediction(image,detectionModel,slice_height=512
            ,slice_width=512,overlap_height_ratio=0.2,overlap_width_ratio=0.2)

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

            with open(predict_dir+'/predictions_list_' + currentmodel + '_' + os.path.basename(imPath) +'.json','w+') as resjson:
                s = json.dumps(result.to_coco_annotations())
                resjson.write(s)
