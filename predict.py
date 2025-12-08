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
from imageUtils import (
boxListEvaluation, boxListEvaluationCentroids, boxesFound, precRecall, maskFromBoxes,
rebuildImageFromTiles, boxCoordsToFile, filter_boxes_by_overlap_and_area_distance,
fillHolesInGrid, borderbox, fileToBoxCoords
)

from torchvision.transforms.functional import to_pil_image

import sys
import time
from PIL import Image, ImageDraw
from tqdm import tqdm

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
    boxPath = os.path.join(conf["TV_dir"],conf["Test_dir"],"labels")
    testImageList = testFileList(testPath)
    predict_dir = conf["Pred_dir"]

    #print(testPath)
    #create predictions dir if it does not exist
    Path(predict_dir).mkdir(parents=True, exist_ok=True)
    dScore = []
    invScore = []
    totalTP, totalFP, totalFN = 0, 0, 0
    
    # Use the prefix parameter directly
    currentmodel = prefix
    
    # Use the same key that train_YOLO uses
    train_res_key = "Train_res" if "Train_res" in conf else "trainResFolder"
    modelpath = os.path.join(conf[train_res_key], "detect", currentmodel, "weights", "best.pt")
    print(f"predict_yolo: Loading model from {modelpath}")
    
    # Verify model exists
    if not Path(modelpath).exists():
        raise FileNotFoundError(f"Model not found at: {modelpath}")
    
    detectionModel = AutoDetectionModel.from_pretrained(
        model_type='yolov8', 
        model_path=modelpath, 
        device=0
    )

    for imPath in testImageList:


        image = cv2.imread(imPath)
        #gtBoxes = fileToBoxCoords(os.path.join(boxPath,os.path.basename(imPath)[:-4]+".txt"),returnCat = False)
        gtBoxes = fileToBoxCoords(os.path.join(boxPath,os.path.basename(imPath)[:-4]+".txt"), returnCat = False, yoloToXYXY=True, imgSize=(image.shape[1], image.shape[0]))

        result = get_sliced_prediction(image,detectionModel,slice_height=512
        ,slice_width=512,overlap_height_ratio=0.2,overlap_width_ratio=0.2,
        verbose = False )

        predBoxes = [ p.bbox.to_xyxy() for p in result.object_prediction_list ] # change from the sahi prediction thing to a list of tuples (int this case, ignore category)
        boxesToTextFile(result,predict_dir+'/predictions_list_' + currentmodel + '_' + os.path.basename(imPath) +'.txt')

        if len(gtBoxes) > 0:
            TP,FP,FN = boxListEvaluation(predBoxes,gtBoxes)
            #update totals
            totalTP += TP
            totalFP += FP
            totalFN += FN

            #print("\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ there were "+str(len(gtBoxes)))
            #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ TP,FP,FN "+str((TP,FP,FN)))


            # at the moment not computing centroid based metrics for yolo
            dS, invS = boxListEvaluationCentroids(predBoxes,gtBoxes)
            dScore.append(dS)
            invScore.append(invS)
        #else:
            #print("predict_yolo found a GT tile wihtout boxes")

        """
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
        """


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

    print("average Precision (centroids) "+str(sum(dScore) / len(dScore)))
    print("average Recall (centroids) "+str(sum(invScore) / len(invScore)))

    prec = 0 if (totalTP+totalFP) == 0 else totalTP/(totalTP+totalFP)
    rec = 0 if (totalTP+totalFN) == 0 else totalTP/(totalTP+totalFN)

    print("global Precision (overlap) "+str(prec))
    print("global Recall (overlap) "+str(rec))

    return sum(dScore) / len(dScore), sum(invScore) / len(invScore) , prec, rec


@torch.no_grad()
def predict_DETR(dataset_test, model, processor, device=None, predConfidence=0.5, 
                 predFolder=None, origFolder=None, max_detections=100):
    """
    DETR inference
  

    """

    print("starting predict_DETR (pytorch-style pipeline)")

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    print("Testing Dataset Length " + str(len(dataset_test)))
    Path(os.path.join(predFolder, "FULL")).mkdir(parents=True, exist_ok=True)

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    totalTP, totalFP, totalFN = 0, 0, 0
    precList, recList = [], []
    dScore, invScore = [], []
    count = 0

    with torch.no_grad():
        for ind, (images, targets) in enumerate(data_loader):
            
            imageName = dataset_test.imageNameList[ind].split(os.sep)[-1]
            
            # Get original image as numpy array
            try:
                imToStore = np.ascontiguousarray(images[0].permute(1, 2, 0).cpu().numpy())
            except Exception:
                imToStore = np.ascontiguousarray(np.array(images[0]))

            # Ensure uint8 for saving
            if imToStore.dtype != np.uint8:
                if imToStore.max() <= 1.0:
                    imToStore = (imToStore * 255).astype(np.uint8)

            orig_height, orig_width = imToStore.shape[:2]

            # Process through DETR processor
            img_for_processor = np.array(imToStore) if not isinstance(imToStore, np.ndarray) else imToStore
            encoding = processor(images=img_for_processor, return_tensors="pt", do_rescale=True).to(device)
            pixel_values = encoding["pixel_values"]
            
            # Get processed dimensions
            batch_size, channels, proc_height, proc_width = pixel_values.shape

            # Run model
            outputs = model(pixel_values=pixel_values)

            # Extract predictions
            logits = outputs.logits[0]  # [num_queries, num_classes+1]
            boxes = outputs.pred_boxes[0]  # [num_queries, 4] in normalized cxcywh
            
            # Convert to probabilities
            # For single-class: logits is [num_queries, 2] where dim=0 is object, dim=1 is no-object
            probs = logits.softmax(-1)
            
            # Get probability of object class (not no-object)
            # For num_labels=1, index 0 is the object class
            scores = probs[:, 0]  # Probability of being an object (class 0)
            labels = torch.zeros_like(scores, dtype=torch.long)  # All predictions are class 0

            # Filter by confidence threshold
            keep = scores > predConfidence
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            # === DIAGNOSTIC (first 5 tiles) ===
            if ind < 5:
                print(f"\n{'='*70}")
                print(f"[DEBUG {ind}] Tile: {imageName}")
                print(f"{'='*70}")
                print(f"Original tile (W×H): {orig_width}×{orig_height}")
                print(f"Processed (W×H): {proc_width}×{proc_height}")
                print(f"Predictions above threshold ({predConfidence}): {keep.sum().item()} / {len(keep)}")
                
                if len(boxes) > 0:
                    print(f"\nFirst 3 predictions (normalized cxcywh):")
                    for i, (box, score, label) in enumerate(zip(boxes[:3], scores[:3], labels[:3])):
                        print(f"  [{i}] box={box.cpu().numpy()}, score={score.item():.3f}, label={label.item()}")

            # ===== COORDINATE TRANSFORMATION =====
            # Step 1: Convert normalized cxcywh to normalized xyxy
            boxes_norm = torch.zeros_like(boxes)
            boxes_norm[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = cx - w/2
            boxes_norm[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = cy - h/2
            boxes_norm[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = cx + w/2
            boxes_norm[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = cy + h/2
            
            # Step 2: Denormalize to processed image coordinates
            boxes_proc = torch.zeros_like(boxes_norm)
            boxes_proc[:, 0] = boxes_norm[:, 0] * proc_width
            boxes_proc[:, 1] = boxes_norm[:, 1] * proc_height
            boxes_proc[:, 2] = boxes_norm[:, 2] * proc_width
            boxes_proc[:, 3] = boxes_norm[:, 3] * proc_height
            
            # Step 3: Scale to original image coordinates
            # The processor maintains aspect ratio, so we need to figure out the scaling
            scale_x = orig_width / proc_width
            scale_y = orig_height / proc_height
            
            boxes_orig = torch.zeros_like(boxes_proc)
            boxes_orig[:, 0] = boxes_proc[:, 0] * scale_x
            boxes_orig[:, 1] = boxes_proc[:, 1] * scale_y
            boxes_orig[:, 2] = boxes_proc[:, 2] * scale_x
            boxes_orig[:, 3] = boxes_proc[:, 3] * scale_y

            if ind < 5:
                print(f"\nCoordinate transformation:")
                print(f"  Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
                if len(boxes_orig) > 0:
                    print(f"\nFirst 3 predictions (original tile xyxy):")
                    for i, box in enumerate(boxes_orig[:3]):
                        x1, y1, x2, y2 = box.cpu().numpy()
                        w, h = x2-x1, y2-y1
                        print(f"  [{i}] [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] (w={w:.1f}, h={h:.1f})")

            # Clip to image boundaries
            boxes_orig[:, 0] = boxes_orig[:, 0].clamp(0, orig_width)
            boxes_orig[:, 1] = boxes_orig[:, 1].clamp(0, orig_height)
            boxes_orig[:, 2] = boxes_orig[:, 2].clamp(0, orig_width)
            boxes_orig[:, 3] = boxes_orig[:, 3].clamp(0, orig_height)

            # Filter out degenerate boxes
            widths = boxes_orig[:, 2] - boxes_orig[:, 0]
            heights = boxes_orig[:, 3] - boxes_orig[:, 1]
            valid = (widths > 1) & (heights > 1)
            
            boxes_orig = boxes_orig[valid]
            labels = labels[valid]
            scores = scores[valid]

            # Apply NMS to remove duplicate detections
            if len(boxes_orig) > 0:
                from torchvision.ops import nms
                # NMS expects boxes in xyxy format (which we have)
                nms_threshold = 0.5  # You can make this a parameter
                keep_nms = nms(boxes_orig, scores, nms_threshold)
                boxes_orig = boxes_orig[keep_nms]
                labels = labels[keep_nms]
                scores = scores[keep_nms]
            
            # Limit number of detections after NMS
            if len(boxes_orig) > max_detections:
                topk = torch.topk(scores, max_detections)
                boxes_orig = boxes_orig[topk.indices]
                labels = labels[topk.indices]
                scores = scores[topk.indices]

            filtered_boxes = boxes_orig.to(cpu_device)
            filtered_labels = labels.to(cpu_device)
            filtered_scores = scores.to(cpu_device)
            
            if ind < 5:
                print(f"After NMS: {len(filtered_boxes)} predictions")

            # Prepare GT boxes in xyxy format
            tile_gt_boxes = []
            if len(targets) > 0 and "annotations" in targets[0]:
                for ann in targets[0]["annotations"]:
                    bx = ann["bbox"]  # [x, y, w, h]
                    x1, y1 = float(bx[0]), float(bx[1])
                    x2, y2 = x1 + float(bx[2]), y1 + float(bx[3])
                    tile_gt_boxes.append([x1, y1, x2, y2])
            
            if ind < 5:
                if len(tile_gt_boxes) > 0:
                    print(f"\nGround truth (original tile xyxy):")
                    for i, box in enumerate(tile_gt_boxes[:3]):
                        x1, y1, x2, y2 = box
                        w, h = x2-x1, y2-y1
                        print(f"  [{i}] [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] (w={w:.1f}, h={h:.1f})")
                else:
                    print(f"\nNo ground truth annotations for this tile")
                print(f"{'='*70}\n")

            gt_boxes_tensor = torch.tensor(tile_gt_boxes, dtype=torch.float32) if tile_gt_boxes else torch.empty((0,4), dtype=torch.float32)

            # Compute metrics if GT exists
            boxCatAndCoords = []
            if gt_boxes_tensor.shape[0] > 0 and len(filtered_boxes) > 0:
                pred_list = [tuple(b.tolist()) for b in filtered_boxes]
                gt_list = [tuple(b) for b in tile_gt_boxes]

                TP, FP, FN = boxListEvaluation(pred_list, gt_list)
                totalTP += TP
                totalFP += FP
                totalFN += FN

                dS, invS = boxListEvaluationCentroids(pred_list, gt_list)
                dScore.append(dS)
                invScore.append(invS)

                thisPrec = 0 if (TP + FP) == 0 else TP/(TP+FP)
                thisRec = 0 if (TP + FN) == 0 else TP/(TP+FN)
                precList.append(thisPrec)
                recList.append(thisRec)

                for lab, box in zip(filtered_labels, filtered_boxes):
                    lab_val = int(lab.tolist()) if isinstance(lab, torch.Tensor) else int(lab)
                    bx = tuple(map(float, box.tolist()))
                    boxCatAndCoords.append((lab_val,) + bx)
            elif gt_boxes_tensor.shape[0] > 0:
                # GT exists but no predictions
                totalFN += len(tile_gt_boxes)

            # Create prediction mask from boxes
            if len(filtered_boxes) > 0:
                predMask = maskFromBoxes([tuple(b.tolist()) for b in filtered_boxes], imToStore.shape)
            else:
                predMask = np.ones((imToStore.shape[0], imToStore.shape[1]), dtype=np.uint8)*255

            # Save outputs
            if predFolder is not None:
                os.makedirs(predFolder, exist_ok=True)
                cv2.imwrite(os.path.join(predFolder, imageName), cv2.cvtColor(imToStore, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(predFolder, "PREDMASK"+imageName), predMask)
                boxCoordsToFile(os.path.join(predFolder, "BOXCOORDS"+imageName[:-4]+".txt"), boxCatAndCoords)

            count += 1

    # Rebuild full images/masks
    for imageN, TileList in dataset_test.getSliceFileInfo().items():
        rebuildImageFromTiles(imageN, TileList, predFolder, origFolder)

    torch.set_num_threads(n_threads)

    # Print metrics
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    if len(dScore) > 0 and len(precList) > 0:
        print(f"Centroid-based Precision: {sum(dScore)/len(dScore):.4f}")
        print(f"Centroid-based Recall: {sum(invScore)/len(invScore):.4f}")
        print(f"Overlap-based Precision (per-tile avg): {sum(precList)/len(precList):.4f}")
        print(f"Overlap-based Recall (per-tile avg): {sum(recList)/len(recList):.4f}")
    else:
        print("No valid tile-level metrics (no GT or predictions matched)")

    prec = 0 if (totalTP + totalFP) == 0 else totalTP/(totalTP+totalFP)
    rec = 0 if (totalTP + totalFN) == 0 else totalTP/(totalTP+totalFN)
    print(f"\nGlobal Overlap Metrics:")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  TP/FP/FN: {totalTP}/{totalFP}/{totalFN}")
    print(f"  Total tiles: {count}")
    print("="*70 + "\n")

    return (sum(dScore)/len(dScore) if dScore else 0,
            sum(invScore)/len(invScore) if invScore else 0,
            prec, rec)


def predict_DeformableDETR_FIXED(dataset_test, model, processor, device=None, 
                                  predConfidence=0.5, predFolder=None, origFolder=None, 
                                  max_detections=100, nms_threshold=0.5):
    """
    Fixed prediction for Deformable DETR
    """
    
    print("Starting Deformable DETR prediction")
    
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Add diagnostic for first batch
    for ind in range(min(1, len(dataset_test))):
        # Get data
        image_data, target = dataset_test[ind]
        imageName = dataset_test.imageNameList[ind].split(os.sep)[-1]
        
        # Convert to proper format
        if isinstance(image_data, torch.Tensor):
            if image_data.shape[0] == 3:  # (C, H, W)
                imToStore = image_data.permute(1, 2, 0).cpu().numpy()
            else:
                imToStore = image_data.cpu().numpy()
        else:
            imToStore = np.array(image_data)
        
        # Ensure uint8
        if imToStore.dtype != np.uint8:
            if imToStore.max() <= 1.0:
                imToStore = (imToStore * 255).astype(np.uint8)
            else:
                imToStore = imToStore.astype(np.uint8)
        
        orig_height, orig_width = imToStore.shape[:2]
        
        # CRITICAL FIX: Convert to PIL Image for the processor
        pil_image = Image.fromarray(imToStore)
        
        # Process - pass PIL Image, not numpy array
        encoding = processor(images=pil_image, return_tensors="pt", do_rescale=True).to(device)
        pixel_values = encoding["pixel_values"]
        
        # Forward pass
        outputs = model(pixel_values=pixel_values)
        
        print("\n" + "="*70)
        print("DEFORMABLE DETR OUTPUT DIAGNOSTIC")
        print("="*70)
        print(f"Image: {imageName}")
        print(f"Original size: {orig_width}×{orig_height}")
        print(f"Processed size: {pixel_values.shape}")
        print(f"Output keys: {outputs.keys()}")
        print(f"Logits shape: {outputs.logits.shape}")
        print(f"Pred boxes shape: {outputs.pred_boxes.shape}")
        
        # Check logits values
        logits = outputs.logits[0]
        print(f"Logits min/max: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        
        # Check probabilities
        probs = logits.softmax(-1)
        print(f"Probs shape: {probs.shape}")
        print(f"Num classes in output: {probs.shape[-1]}")
        
        # For single-class detection with num_labels=1:
        # probs should be shape [num_queries, 2] where:
        # - probs[:, 0] = probability of object (class 0)
        # - probs[:, 1] = probability of no-object
        
        if probs.shape[-1] == 2:
            # Correct: single class + no-object
            scores_class0 = probs[:, 0]
            print(f"✓ Single-class mode detected (correct: 2 output dims)")
        elif probs.shape[-1] == 1:
            # Wrong: only one output - this is the bug!
            print(f"✗ ERROR: Only 1 output dimension! Model needs retraining with fixed config")
            print(f"   Current model outputs {probs.shape[-1]} class, should be 2 (object + no-object)")
            print(f"   All scores will be 1.0 due to softmax on single value")
            print(f"   SOLUTION: Delete the .pth file and retrain with the fixed training code")
            scores_class0 = probs[:, 0]
        else:
            # Multi-class (shouldn't happen with num_labels=1)
            print(f"WARNING: Expected 2 classes (object/no-object) but got {probs.shape[-1]}")
            scores_class0 = probs[:, 0]
        
        print(f"Class 0 scores: min={scores_class0.min().item():.4f}, max={scores_class0.max().item():.4f}")
        print(f"Predictions with score>0.5: {(scores_class0 > 0.5).sum().item()}")
        print(f"Predictions with score>0.9: {(scores_class0 > 0.9).sum().item()}")
        print(f"Predictions with score>0.99: {(scores_class0 > 0.99).sum().item()}")
        
        # Check predicted boxes
        boxes = outputs.pred_boxes[0]
        print(f"Boxes (first 5): {boxes[:5].tolist()}")
        print("="*70 + "\n")
        
        break

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
def predict_pytorch(dataset_test, model, device, predConfidence, postProcess, predFolder, origFolder):
    """
        Inference for pytorch object detectors
        possible postprocess values, 0 for no postprocess
        1 for only double box removal
        2 for double box and grid filling
    """
    def boxAndCatsToList(lab, boxes):
        """
            receive tensors with labels and categories an translate them to lists of boxes
        """
        retList =[]
        # create a new list of tuples with category predictions to save to file
        for el,tup in zip(lab, boxes):
            # convert so they are not tensors
            el = el.tolist()
            tup = tuple(tup.tolist())
            retList.append((el,)+tup)
        return retList

    data_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    print("Testing Dataset Length "+str(len(dataset_test)))
    #print("testing dataset image dict "+str(dataset_test.slicesToImages))

    # create output folder if necessary
    Path(os.path.join(predFolder,"FULL")).mkdir(parents=True, exist_ok=True)

    # evaluate on the test dataset (why???)
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    #print("evaluating "+str(len(data_loader)))

    count=0
    precList = []
    recList = []
    dScore = []
    invScore = []
    ignoreCount = 0

    totalTP, totalFP, totalFN = 0,0,0
    with torch.no_grad():
        for ind, (images, targets) in enumerate(data_loader):
            imageName = dataset_test.imageNameList[ind].split(os.sep)[-1]
            #print("testing "+str(imageName))
            imToStore = np.ascontiguousarray(images[0].permute(1, 2, 0))
            height, width = imToStore.shape[:2]

            # store test images to disk
            images = list(img.to(device) for img in images)

            torch.cuda.synchronize()
            outputs = model(images)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            # Move outputs to CPU and filter based on confidence score, also filter out border boxes.
            filtered_outputs = []
            for output in outputs:
                high_conf_indices = output['scores'] > predConfidence

                keep_indices = []
                for i, box in enumerate(output['boxes']):
                    if high_conf_indices[i] and not borderbox(box, width, height):
                        keep_indices.append(i)

                keep_indices = torch.tensor(keep_indices, dtype=torch.long)

                filtered_output = {
                    'boxes': output['boxes'][keep_indices].to(cpu_device),
                    'labels': output['labels'][keep_indices].to(cpu_device),
                    'scores': output['scores'][keep_indices].to(cpu_device),
                }
                filtered_outputs.append(filtered_output)

            # filter out border boxes for targets
            boxes = targets[0]['boxes']
            keep_target_indices = [] if len(boxes) == 0 else [
                i for i, box in enumerate(boxes) if not borderbox(box, width, height)
            ]
            if keep_target_indices:
                keep_target_indices = torch.tensor(keep_target_indices, dtype=torch.long)
                targets[0]['boxes'] = boxes[keep_target_indices]
                targets[0]['labels'] = targets[0]['labels'][keep_target_indices]
            else:
                targets[0]['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                targets[0]['labels'] = torch.empty((0,), dtype=targets[0]['labels'].dtype)


            #if len(filtered_outputs[0]["boxes"]) >= 0 :
            if len(targets[0]['boxes']) > 0: # ignore tiles without boxes
                #print("this image has GT boxes "+str(imageName))
                correctedLabels, correctedBoxes = filtered_outputs[0]["labels"],filtered_outputs[0]["boxes"]
                # apply postprocessing if needed
                # modify boxes and labels to eliminate boxes with too much overlap
                if postProcess > 0: correctedLabels, correctedBoxes = filter_boxes_by_overlap_and_area_distance(filtered_outputs[0]["labels"],filtered_outputs[0]["boxes"], 255*imToStore, os.path.join(predFolder,"removal"+imageName))

                # now try to fill holes in the grid
                if postProcess > 1: correctedLabels, correctedBoxes = fillHolesInGrid(correctedLabels, correctedBoxes, imToStore, os.path.join(predFolder,"newBoxes"+imageName))

                TP,FP,FN = boxListEvaluation(correctedBoxes,targets[0]["boxes"])
                #update totals
                totalTP += TP
                totalFP += FP
                totalFN += FN
                dS, invS = boxListEvaluationCentroids(correctedBoxes,targets[0]["boxes"])
                boxCoords = correctedBoxes
                #boxCatAndCoords = boxAndCatsToList(correctedLabels, correctedBoxes)

                #print("corrected boxes")
                #print(correctedBoxes)
                #print("targets")
                #print(targets[0]["boxes"])


                boxCatAndCoords = []

                # create a new list of tuples with category predictions to save to file
                for el,tup in zip(correctedLabels, correctedBoxes):
                    # convert so they are not tensors
                    el = el.tolist()
                    tup = tuple(tup.tolist())
                    boxCatAndCoords.append((el,)+tup)

                thisPrec = 0 if (TP+FP) == 0 else (TP/(TP+FP))
                precList.append(thisPrec)
                thisRec = 0 if (TP+FN) == 0 else (TP/(TP+FN))
                recList.append(thisRec)
                dScore.append(dS)
                invScore.append(invS)
                #print("for image "+str(imageName)+" got "+str((thisPrec,thisRec)))

                # store image, predicted mask and box coords
                predMask = maskFromBoxes(boxCoords,imToStore.shape)
                #print("writing predmask "+str(os.path.join(predFolder,"PREDMASK"+imageName)))
                cv2.imwrite( os.path.join(predFolder,"PREDMASK"+imageName),predMask  )
                boxCoordsToFile(os.path.join(predFolder,"BOXCOORDS"+imageName[:-4]+".txt"),boxCatAndCoords)

                #if len(targets[0]['boxes']) >120:
                #    cv2.imwrite( str(len(targets[0]['boxes']))+"boxes"+imageName, imToStore*255 )
                #    boxCoordsToFile("BOXCOORDS"+imageName[:-4]+".txt",boxCatAndCoords)

                count+=1


            #else:
                #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ this image has no GT boxes "+str(imageName))

                # ignore if there are no outputs unless there should be
            #    if len(targets[0]['boxes']) >= 5:
            #        prec,rec, dS, invS , boxCoords, boxCatAndCoords = 0, 0, 0, 0, [], []
            #        print("in this case I did not predict boxes and I should have "+str(len(targets[0]['boxes']))+" but i only had "+str(len(filtered_outputs[0]["boxes"])))


            #        precList.append(prec)
            #        recList.append(rec)
            #        dScore.append(dS)
            #        invScore.append(invS)

            # always store the tile
            cv2.imwrite( os.path.join(predFolder,imageName), imToStore*255 )

            # clean up    
            del outputs
            del filtered_outputs
            torch.cuda.empty_cache()


    # now reconstruct the full images and masks from what we have in the folder
    # the original data folder should also be accessed and passed to the function
    for imageN,TileList  in dataset_test.getSliceFileInfo().items():
        rebuildImageFromTiles(imageN, TileList, predFolder, origFolder)

    # accumulate predictions from all images
    torch.set_num_threads(n_threads)

    # computations
    #print(invScore)
    #print(dScore)
    print("average Precision (centroids) "+str(sum(dScore) / len(dScore)))
    print("average Recall (centroids) "+str(sum(invScore) / len(invScore)))

    print(precList)
    print(recList)
    print("average Precision (overlap) "+str(sum(precList) / len(precList)))
    print("average Recall (overlap) "+str(sum(recList) / len(recList)))

    # trying to postprocess from the full image is inneficcient because there are too many boxes to check at the same time
    if sum(precList) / len(precList) >1: raise Exception("what is this shit?")

    # compute global precision and recall
    prec = 0 if (totalTP+totalFP) == 0 else totalTP/(totalTP+totalFP)
    rec = 0 if (totalTP+totalFN) == 0 else totalTP/(totalTP+totalFN)
    print("global Precision (overlap) "+str(prec))
    print("global Recall (overlap) "+str(rec))


    #return sum(dScore) / len(dScore), sum(invScore) / len(invScore) ,sum(precList) / len(precList), sum(recList) / len(recList)
    return sum(dScore) / len(dScore), sum(invScore) / len(invScore) , prec, rec
