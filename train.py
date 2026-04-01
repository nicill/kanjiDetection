import os
from ultralytics import YOLO
from ultralytics import settings
import torch
import yaml
import sys
import shutil

from datasets import ODDataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T

from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils, MaskRCNN
from torchvision.models.detection import SSD300_VGG16_Weights

from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models.detection.backbone_utils import BackboneWithFPN, LastLevelMaxPool

from functools import partial
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.fcos import FCOSClassificationHead

from utils import MetricLogger,SmoothedValue,warmup_lr_scheduler,reduce_dict
import math
import cv2
import numpy as np

import torch.nn as nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import (convnext_tiny, convnext_small, convnext_base, convnext_large, ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights, ConvNeXt_Large_Weights)

from torchvision.models.detection.rpn import AnchorGenerator

from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig

import time
#from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR

import multiprocessing


torch.backends.cudnn.benchmark = True

# Channel sizes for each ConvNeXt stage
input_channels_dict = {
    'convnext_tiny':   [96, 192, 384, 768],
    'convnext_small':  [96, 192, 384, 768],
    'convnext_base':   [128, 256, 512, 1024],
    'convnext_large':  [192, 384, 768, 1536],
}


class BackboneWithFPN(nn.Module):
    def __init__(
        self, backbone, in_channels_list, out_channels,
        extra_blocks=None, norm_layer=None
    ):
        super().__init__()
        self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks or LastLevelMaxPool(),
            norm_layer=norm_layer
        )
        self.out_channels = out_channels

    def forward(self, x):
        return self.fpn(self.body(x))


def convnext_fpn_backbone(backbone_name='convnext_base', trainable_layers=8, extra_blocks=None, norm_layer=None, out_channels=256):
    # Map name to model + weights
    backbone_fns = {
        "convnext_tiny":   (convnext_tiny,   ConvNeXt_Tiny_Weights.DEFAULT),
        "convnext_small":  (convnext_small,  ConvNeXt_Small_Weights.DEFAULT),
        "convnext_base":   (convnext_base,   ConvNeXt_Base_Weights.DEFAULT),
        "convnext_large":  (convnext_large,  ConvNeXt_Large_Weights.DEFAULT),
    }

    if backbone_name not in backbone_fns:
        raise ValueError(f"Invalid backbone name: {backbone_name}")

    backbone_fn, weights = backbone_fns[backbone_name]
    channels = input_channels_dict[backbone_name]

    # Get ConvNeXt features and create extractor
    convnext = backbone_fn(weights=weights).features
    return_layers = {'1': '0', '3': '1', '5': '2', '7': '3'}
    body = create_feature_extractor(convnext, return_layers)

    # Freeze layers
    if not (0 <= trainable_layers <= 8):
        raise ValueError("trainable_layers must be between 0 and 8")
    freeze = [name for name, _ in body.named_parameters()]
    if trainable_layers < 8:
        unfrozen = ["1", "3", "5", "7"][-trainable_layers:]
        freeze = [n for n in freeze if all(not n.startswith(u) for u in unfrozen)]
    for name, param in body.named_parameters():
        if name in freeze:
            param.requires_grad_(False)

    return BackboneWithFPN(backbone=body, in_channels_list=channels, out_channels=out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer)


def makeTrainYAML(conf, fileName = 'trainAUTO.yaml', pDict = {}):
    """
    Function to write a yaml file
    """
    data = { "names": {0: 'Kanji'}, "path": conf["TV_dir"], "train": conf["Train_dir"],
            "val": conf["Valid_dir"]}
    with open(fileName, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def train_YOLO(conf, datasrc, prefix = 'combined_data_', params = {}):
    """
        train a yolo model with LR, mosaic and score as parameters
    """
    resfolder = conf["Train_res"]
    print("")
    epochs = conf["ep"]
    name = prefix+"epochs"+str(epochs)+'ex'
    valfolder = conf["Valid_res"]
    imgsize = conf["slice"]
    settings.update({'runs_dir':resfolder})
    model = YOLO('yolov8n.pt')
    scVal = 0.5 if "scale" not in params else params["scale"]
    mosVal = 1.0 if "mosaic" not in params else params["mosaic"]
    lr0Val = 0.01 if "lr0" not in params else params["lr0"]
    
    overrides = {'data': datasrc, 'epochs': epochs, 'imgsz': imgsize, 'name': name, 'device': 0, 'patience': 10, 'exist_ok': True, 'optimizer': 'SGD', 'scale': scVal, 'mosaic': mosVal,
                 'fliplr': 0.0, 'flipud': 0.0, 'box': 10.0, 'cls': 0.3, 'lr0': lr0Val, 'project': resfolder,}
    train_results = model.train(**overrides)
    # check overrides were correctly read
    print("@@@ TRAINING ARGS CHECK @@@")
    for k in ['fliplr', 'flipud', 'box', 'cls', 'scale', 'mosaic', 'epochs', 'imgsz', 'lr0']:
        print(f"  {k}: {getattr(model.trainer.args, k, 'NOT FOUND')}")
    model.val(project=resfolder, name=valfolder, save_json=True)
    model_path = os.path.join(train_results.save_dir,"weights","best.pt")
    print(f"Model saved to: {model_path}")

    file_path = ("expYOLO"+"LR" +str(params["lr0"])+"scale"+str(params["scale"])+"mosaic"+str(params["mosaic"])+"Epochs" + str(conf["ep"]) + ".pt" )
    shutil.copy(model_path, file_path)

    return file_path

def train_DETR(conf, datasrc, prefix='detr_exp_', params=None, file_path=""):
    """
    Train DETR model with correct configuration for single-class detection.
    """
    params = params or {}
    epochs = params.get("num_epochs", conf.get("ep", 10))
    processor_name = params.get("processor_name", "facebook/detr-resnet-50")
    trainAgain = params.get("trainAgain", True)
    device = params.get("device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    batch_size = params.get("batch_size", 1)
    lr = params.get("lr", 1e-5)
    weight_decay = params.get("weight_decay", 1e-4)
    
    print(f"[DETR] File: {file_path}, Train: {trainAgain}, Device: {device}, Epochs: {epochs}")
    
    processor = DetrImageProcessor.from_pretrained(processor_name)
    
    # Create or load model
    config = DetrConfig.from_pretrained(processor_name)
    config.num_labels = 1
    model = DetrForObjectDetection(config)
    
    if trainAgain:
        # Initialize from pretrained (except classification head)
        print(f"[DETR] Configuring model with num_labels={config.num_labels}")
        base_model = DetrForObjectDetection.from_pretrained(processor_name)
        
        pretrained_dict = {k: v for k, v in base_model.state_dict().items() 
                          if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
        
        model.load_state_dict(pretrained_dict, strict=False)
        print(f"[DETR] Initialized with pretrained weights (except class head)")
    elif os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path, map_location='cpu'))
        print(f"[DETR] Loaded model from {file_path}")
    
    model.to(device)
    
    if not trainAgain:
        model.eval()
        return model
    
    # Training setup
    data_loader = torch.utils.data.DataLoader(
        datasrc, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_DETR_skip_empty
    )
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay
    )
    
    # Learning rate schedule
    step_size = max(epochs // 4 if epochs >= 100 else epochs // 3, 5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    print(f"[DETR] LR schedule: drop by 0.1x every {step_size} epochs")
    
    # Training loop
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(data_loader):
            # Prepare data
            annotations = [{"image_id": i, "annotations": t["annotations"]} 
                          for i, t in enumerate(targets)]
            
            encoding = processor(images=list(images), annotations=annotations, 
                               return_tensors="pt", do_rescale=True)
            
            pixel_values = encoding["pixel_values"].to(device)
            hf_labels = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in lab.items()} for lab in encoding["labels"]]
            
            # Diagnostic (first batch only)
            if epoch == 0 and batch_idx == 0:
                print("\n" + "="*70)
                print("TRAINING DIAGNOSTIC")
                print("="*70)
                print(f"Config: num_labels={model.config.num_labels}, batch_size={len(images)}")
                print(f"Pixel values: {pixel_values.shape}")
                if hf_labels:
                    print(f"First target: boxes={hf_labels[0]['boxes'].shape}, "
                          f"classes={hf_labels[0]['class_labels'].unique().tolist()}")
                print("="*70 + "\n")
            
            # Forward + backward
            outputs = model(pixel_values=pixel_values, labels=hf_labels)
            loss = outputs.loss if outputs.loss is not None else sum(outputs.loss_dict.values())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        lr_scheduler.step()
        avg_loss = total_loss / len(data_loader)
        
        print(f"[DETR] Epoch {epoch+1}/{epochs} - loss: {avg_loss:.6f}")
        if hasattr(outputs, 'loss_dict'):
            print(f"  {', '.join(f'{k}: {v.item():.4f}' for k, v in outputs.loss_dict.items())}")
    
    torch.save(model.state_dict(), file_path)
    print(f"[DETR] Saved to {file_path}")
    
    return model

def get_maskrcnn_convnext(num_classes, backbone_name='convnext_base', v2=True, min_size=224, max_size=1333, input_channels_dict=None, backbone_fns=None, extra_blocks=None, norm_layer=None, out_channels=256, trainable_layers=3,):
    if input_channels_dict is None:
        input_channels_dict = {
            'convnext_tiny': [96, 192, 384, 768],
            'convnext_small': [96, 192, 384, 768],
            'convnext_base': [128, 256, 512, 1024],
            'convnext_large': [192, 384, 768, 1536],
        }
    # Map name to model + weights
    if backbone_fns is None:
        backbone_fns = {
            "convnext_tiny":   (convnext_tiny,   ConvNeXt_Tiny_Weights.DEFAULT),
            "convnext_small":  (convnext_small,  ConvNeXt_Small_Weights.DEFAULT),
            "convnext_base":   (convnext_base,   ConvNeXt_Base_Weights.DEFAULT),
            "convnext_large":  (convnext_large,  ConvNeXt_Large_Weights.DEFAULT),
        }

    if backbone_name not in backbone_fns:
        raise ValueError(f"Invalid backbone name: {backbone_name}")

    backbone_fn, weights = backbone_fns[backbone_name]
    channels = input_channels_dict[backbone_name]

    # Get ConvNeXt features and create extractor
    convnext = backbone_fn(weights=weights).features
    return_layers = {'1': '0', '3': '1', '5': '2', '7': '3'}

    # select layers that won't be frozen
    if trainable_layers < 0 or trainable_layers > 8:
        raise ValueError(f"Trainable layers should be in the range [0,8], got {trainable_layers}")
    layers_to_train = ["7", "6", "5", "4", "3", "2", "1", "0"][:trainable_layers]
    for name, parameter in convnext.named_parameters():
        if not name[0] in layers_to_train:
            parameter.requires_grad_(False)

    backbone =  BackboneWithFPN(
        backbone=convnext,
        return_layers=return_layers,
        in_channels_list=channels,
        out_channels=out_channels,
        extra_blocks=extra_blocks,
        norm_layer=norm_layer
    )

    if v2:
        rpn_anchor_generator = _default_anchorgen()
        rpn_head = RPNHead(
            backbone.out_channels,
            rpn_anchor_generator.num_anchors_per_location()[0],
            conv_depth=2
        )
        box_head = FastRCNNConvFCHead(
            (backbone.out_channels, 7, 7),
            [256, 256, 256, 256],
            [1024],
            norm_layer=nn.BatchNorm2d
        )
        mask_head = MaskRCNNHeads(
            backbone.out_channels,
            [256, 256, 256, 256],
            1,
            norm_layer=nn.BatchNorm2d
        )
        model = MaskRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            box_head=box_head,
            mask_head=mask_head,
            min_size=min_size,
            max_size=max_size
        )
    else:
        model = MaskRCNN(
            backbone,
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size
        )

    # Replace box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )

    return model

"""
def get_maskrcnn_convnext(num_classes, backbone_name='convnext_base', min_size=2048, max_size=2048):
    backbone = convnext_fpn_backbone(backbone_name=backbone_name, trainable_layers=8)
    anchor_generator = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)), aspect_ratios=((0.75, 1.0, 1.5),) * 5)
    model = MaskRCNN(backbone, num_classes=num_classes, min_size=min_size, max_size=max_size, rpn_anchor_generator=anchor_generator)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model
"""

def train_pytorchModel(dataset, device, num_classes, file_path, num_epochs = 10, trainAgain = False, proportion = 0.9, BS = 1, mType = "maskrcnn", trainParams = {"modelType": "maskrcnn", "score": 0.05, "nms": 0.25, "predconf": 0.7, "LR": 0.005, "STEP": 100, "GAMMA":0.1}):

    #num_workers = min(8, max(1, multiprocessing.cpu_count() - 4))  # tune if needed
    #num_workers = 2 
    #data_loader = torch.utils.data.DataLoader(dataset, batch_size = BS, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=False, persistent_workers=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BS, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=False)

    # probably have a look at this  https://discuss.pytorch.org/t/how-to-use-collate-fn/27181/2
    print("Training Dataset Length "+str(len(dataset)))

    if trainAgain :
        model = get_model_instance_segmentation(num_classes,mType = mType)
        if mType in ["maskrcnn","fasterrcnn","convnextmaskrcnn"]:
            model.roi_heads.score_thresh = trainParams["score"]  # Increase to filter out low-confidence boxes (default ~0.05)
            model.roi_heads.nms_thresh = trainParams["nms"]   # Reduce to suppress more overlapping boxes (default ~0.5)
        elif mType in ["retinanet","fcos","ssd"]:
            model.score_thresh = trainParams["score"]
            model.nms_thresh = trainParams["nms"]

        # move model to the right device
        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        #optimizer = torch.optim.SGD(params, lr = 0.005, momentum=0.9, weight_decay=0.0005)
        optimizer = torch.optim.SGD(params, lr = trainParams["LR"], momentum=0.9, weight_decay=0.0005)

        # and a learning rate scheduler
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1 )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = trainParams["STEP"], gamma = trainParams["GAMMA"] )

        # train with patience
        patience = 10
        best_loss = 9999
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            epoch_loss = metric_logger.loss.global_avg
            # update the learning rate
            lr_scheduler.step()

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), file_path)  # save best to final path
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}, best loss {best_loss:.4f}")
                    break
            print("@@@@@@@@@@@@@@@@@@@@ EPOCH LOSS "+str(epoch_loss)+" BEST LOSS "+str(best_loss))

        # Save the model's state_dict 
        model.load_state_dict(torch.load(file_path))
        model.to(device)

    else:# loading pretrained model
        print("not training again")
        model = get_model_instance_segmentation(num_classes, mType=mType)

        #reload the model
        model.to(device)

        # Load the saved state_dict into the model
        model.load_state_dict(torch.load(file_path))

        # update heads and shit
        if mType in ["maskrcnn","fasterrcnn","convnextmaskrcnn"]:
            model.roi_heads.score_thresh = trainParams["score"]
            model.roi_heads.nms_thresh = trainParams["nms"]
        elif mType in ["retinanet","fcos","ssd"]:
            model.score_thresh = trainParams["score"]
            model.nms_thresh = trainParams["nms"]
        
        model.eval()  # Set the model to evaluation mode
    print("finished training")
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        try:

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                raise Exception("NAN LOSS in train one epoch")

            optimizer.zero_grad()
            losses.backward()
            # gradient clipping!
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM on batch — skipping. Consider reducing BS.")
                optimizer.zero_grad()
            else:
                raise e
        finally:
            for var in ['images', 'targets', 'loss_dict', 'losses', 'loss_dict_reduced', 'losses_reduced']:
                if var in dir():
                    del var
            torch.cuda.empty_cache()
 
    return metric_logger

def collate_fn(batch):
    return tuple(zip(*batch))


# Alternative: If you want to skip tiles with no annotations during training
def collate_fn_DETR_skip_empty(batch):
    """
    Collate function that SKIPS samples with no annotations during training.
    Use this if you want to avoid training on empty tiles.
    """
    filtered_batch = []
    
    for image, target in batch:
        # Skip if no annotations
        if "annotations" in target and len(target["annotations"]) > 0:
            filtered_batch.append((image, target))
    
    # If all samples were empty, return a dummy batch
    # (This shouldn't happen often with proper dataset filtering)
    if len(filtered_batch) == 0:
        print("WARNING: Empty batch after filtering, using first sample from original batch")
        filtered_batch = [batch[0]]
    
    images = []
    targets = []
    
    for image, target in filtered_batch:
        if isinstance(image, torch.Tensor):
            images.append(image)
        else:
            if isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[2] in [1, 3]:
                    image = torch.from_numpy(image).permute(2, 0, 1)
                else:
                    image = torch.from_numpy(image)
            images.append(image)
        targets.append(target)
    
    return images, targets


def get_model_instance_segmentation(num_classes, mType = "maskrcnn"):
    # load an instance segmentation model pre-trained on COCO
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    if mType == "maskrcnn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one (this is only to change the number of classes predicted)
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one (this is only to change the number of classes predicted)
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    elif mType == "fasterrcnn":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one (this is only to change the number of classes predicted)
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    elif mType == "retinanet":
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
    elif mType == "convnextmaskrcnn":
        model = get_maskrcnn_convnext(num_classes = num_classes)
    elif mType == "fcos":
        model = torchvision.models.detection.fcos_resnet50_fpn(
        weights='DEFAULT')
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = FCOSClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
        min_size=640
        max_size=640
        model.transform.min_size = (min_size, )
        model.transform.max_size = max_size
        for param in model.parameters():
            param.requires_grad = True
    elif mType == "ssd":
        size = 300
        # Load the Torchvision pretrained model.
        model = torchvision.models.detection.ssd300_vgg16(
            weights=SSD300_VGG16_Weights.COCO_V1
        )
        # Retrieve the list of input channels.
        in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
        # List containing number of anchors based on aspect ratios.
        num_anchors = model.anchor_generator.num_anchors_per_location()
        # The classification head.
        model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )
        # Image size for transforms.
        model.transform.min_size = (size,)
        model.transform.max_size = size
    else: raise Exception(" get_model_instance_segmentation, unrecognized model type")

    # to change the model, see, for example
    # https://github.com/pytorch/tutorials/blob/d686b662932a380a58b7683425faa00c06bcf502/intermediate_source/torchvision_tutorial.rst



    return model

def get_transform():
    transforms = []
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

if __name__ == '__main__':
    num_epochs = 5
    train_pytorchModel(sys.argv[1],num_epochs)
