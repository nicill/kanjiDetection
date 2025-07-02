import os
from ultralytics import YOLO
from ultralytics import settings
import torch
import yaml
import sys

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
from torchvision.models import (
    convnext_tiny, convnext_small, convnext_base, convnext_large,
    ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights)


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


def convnext_fpn_backbone(
    backbone_name='convnext_base',
    trainable_layers=6,
    extra_blocks=None,
    norm_layer=None,
    out_channels=256
):
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

    return BackboneWithFPN(
        backbone=body,
        in_channels_list=channels,
        out_channels=out_channels,
        extra_blocks=extra_blocks,
        norm_layer=norm_layer
    )


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
    Train a YOlO model from a
    train and validation folder
    """
    resfolder = conf["Train_res"]
    epochs = conf["ep"]
    name = prefix+"epochs"+str(epochs)+'ex'
    valfolder = conf["Valid_res"]
    imgsize = conf["slice"]
    resultstxt = os.path.join(resfolder,valfolder,'results_' + name + '.txt')

    settings.update({'runs_dir':resfolder})
    model = YOLO('yolov8n.pt')

    # now also add any possible parameters from an experiment
    scVal = 0.5 if "scale" not in params else params["scale"]
    mosVal = 1.0 if "mosaic" not in params else params["mosaic"]

    overrides = {'data': datasrc, 'epochs': epochs, 'imgsz': imgsize, 'name': name, 'device': 0,
    'patience': 10,'exist_ok': True, 'scale': scVal, 'mosaic': mosVal}

    # this change has not been tested!
    results = model.train(**overrides)

    #results = model.train(data=datasrc, epochs=epochs, imgsz=imgsize,
    #            name=name,device=0, patience = 10, exist_ok = True,
    #            scale = scVal, mosaic = mosVal)
    results = model.val(project=resfolder,name=valfolder,save_json=True)
    # fuck it, not writing the results to disk

def get_maskrcnn_convnext(num_classes, backbone_name='convnext_base', min_size=224, max_size=1333):
    backbone = convnext_fpn_backbone(backbone_name=backbone_name)
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
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model


def train_pytorchModel(dataset, device, num_classes, file_path, num_epochs = 10,
                        trainAgain = False, proportion = 0.9, mType = "maskrcnn",
                        trainParams = {"score":0.5,"nms":0.3} ):

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )

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
        optimizer = torch.optim.SGD(
            params,
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005
        )

        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )

        # train
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()

        # Save the model's state_dict (recommended for saving models)
        torch.save(model.state_dict(), file_path)
    else:# loading pretrained model
        print("not training again")
        if mType == "maskrcnn":
            model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
            # Replace the box predictor with one matching the saved model's class count (2 classes, including background)
            # get number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            # Reapply the mask predictor
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256  # Ensure this matches your training setup
            model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask,
                hidden_layer,
                num_classes
            )
        elif mType == "fasterrcnn":
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")

            # get number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one (this is only to change the number of classes predicted)
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        elif mType == "convnextmaskrcnn":
            backbone = convnext_fpn_backbone() # use defaults convnext base and 6 trainable layers

            model = MaskRCNN(
                backbone,
                num_classes=num_classes,
                min_size=224,  # match what you trained with
                max_size=1333,
            )

            # Step 2: Replace the heads (optional, for sanity check)
            # Box predictor
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            # Mask predictor
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        elif mType == "retinanet":
            model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
                weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1


            )
            num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head = RetinaNetClassificationHead(
                in_channels=256,
                num_anchors=num_anchors,
                num_classes=num_classes,
                norm_layer=partial(torch.nn.GroupNorm, 32)
            )
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
        else: raise Exception(" train_pytorchModel, unrecognized model type "+str(mType))

        #reload the model
        model.to(device)

        # Load the saved state_dict into the model
        model.load_state_dict(torch.load(file_path))
        model.eval()  # Set the model to evaluation mode
    return model
    print("finished training")

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
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def collate_fn(batch):
    return tuple(zip(*batch))

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
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
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
