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

from transformers import DetrForObjectDetection, DetrImageProcessor
import time


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

    #print("yolo printing parameters BEFORE!!!!!")
    #print(overrides)

    #results = model.train(**overrides)
    # trying to get yolo to understand the parameter changes
    model.overrides.update(overrides)
    results = model.train()

    #print("yolo printing parameters")
    #print(overrides)
    #print(model.trainer.args)

    #results = model.train(data=datasrc, epochs=epochs, imgsz=imgsize,
    #            name=name,device=0, patience = 10, exist_ok = True,
    #            scale = scVal, mosaic = mosVal)
    results = model.val(project=resfolder,name=valfolder,save_json=True)
    # fuck it, not writing the results to disk


def diagnose_training_data(data_loader, processor, num_samples=3):
    """
    Diagnose training data format issues
    """
    print("\n" + "="*60)
    print("DIAGNOSTIC: Checking Training Data Format")
    print("="*60)
    
    for i, (images, targets) in enumerate(data_loader):
        if i >= num_samples:
            break
            
        print(f"\n--- Sample {i} ---")
        
        # Get original image
        try:
            img_np = images[0].permute(1, 2, 0).cpu().numpy()
        except:
            img_np = np.array(images[0])
        
        orig_h, orig_w = img_np.shape[:2]
        print(f"Original image shape (HxW): {orig_h}x{orig_w}")
        
        # Get annotations BEFORE processor
        if len(targets) > 0 and "annotations" in targets[0]:
            raw_anns = targets[0]["annotations"]
            print(f"Number of annotations: {len(raw_anns)}")
            
            # Check first few annotations
            for j, ann in enumerate(raw_anns[:3]):
                bbox = ann["bbox"]
                print(f"  Ann {j}: bbox={bbox}, category={ann.get('category_id', 'N/A')}")
                
                # Verify bbox is valid
                x, y, w, h = bbox
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    print(f"    ⚠️ WARNING: Invalid bbox coordinates!")
                if x + w > orig_w or y + h > orig_h:
                    print(f"    ⚠️ WARNING: Bbox extends beyond image! (img: {orig_w}x{orig_h})")
        
        # Process through processor
        imgs = list(images)
        annotations = [{"image_id": idx, "annotations": t["annotations"]} 
                      for idx, t in enumerate(targets)]
        
        encoding = processor(images=imgs, annotations=annotations, 
                           return_tensors="pt", do_rescale=True)
        
        pixel_values = encoding["pixel_values"]
        proc_h, proc_w = pixel_values.shape[2], pixel_values.shape[3]
        print(f"Processed image shape (HxW): {proc_h}x{proc_w}")
        
        # Check processed labels
        labels = encoding["labels"]
        if len(labels) > 0:
            label = labels[0]
            print(f"Processed label keys: {label.keys()}")
            
            if "boxes" in label:
                boxes = label["boxes"]
                print(f"  Boxes shape: {boxes.shape}")
                print(f"  Boxes (first 3): {boxes[:3].tolist()}")
                print(f"  Boxes format: {'normalized cxcywh' if boxes.max() <= 1.0 else 'absolute coordinates'}")
                
                # Check if boxes are reasonable
                if boxes.max() > 1.0:
                    print(f"    ⚠️ WARNING: Boxes should be normalized [0,1] but found max={boxes.max():.4f}")
            
            if "class_labels" in label:
                class_labels = label["class_labels"]
                print(f"  Class labels: {class_labels.tolist()}")
        
        print(f"  Pixel values range: [{pixel_values.min():.4f}, {pixel_values.max():.4f}]")
    
    print("\n" + "="*60)
    print("End of Diagnostic")
    print("="*60 + "\n")


def train_DETR(conf, datasrc, prefix='detr_exp_', params=None, file_path=""):
    """
    Train DETR model with correct configuration for single-class detection.
    
    CRITICAL: Must configure model for num_labels=1 (single class) before training.
    """
    params = {} if params is None else params
    epochs = params.get("num_epochs", conf.get("ep", 10))
    processor_name = params.get("processor_name", "facebook/detr-resnet-50")
    trainAgain = params.get("trainAgain", True)
    device = params.get("device", torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    batch_size = params.get("batch_size", 1)
    lr = params.get("lr", 1e-5)
    weight_decay = params.get("weight_decay", 1e-4)
    
    print(f"[DETR] results -> {file_path}, trainAgain={trainAgain}, device={device}, epochs={epochs}")
    
    processor = DetrImageProcessor.from_pretrained(processor_name)
    
    if trainAgain:
        # CRITICAL FIX: Configure model for single-class detection
        # Load base model
        base_model = DetrForObjectDetection.from_pretrained(processor_name)
        
        # Get the original config and modify it
        config = base_model.config
        
        # IMPORTANT: Set num_labels to 1 (single object class, not counting background)
        # The model internally handles the "no-object" class
        config.num_labels = 1
        
        print(f"[DETR] Configuring model with num_labels={config.num_labels}")
        
        # Create new model with correct configuration
        from transformers import DetrConfig
        model = DetrForObjectDetection(config)
        
        # Optionally: Initialize from pretrained weights (except the classification head)
        # This helps with faster convergence
        pretrained_dict = base_model.state_dict()
        model_dict = model.state_dict()
        
        # Filter out classification head weights (they have wrong dimensions)
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
        
        # Update model with compatible weights
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f"[DETR] Initialized model with pretrained weights (except class head)")
    else:
        # Load trained model
        config = DetrConfig.from_pretrained(processor_name)
        config.num_labels = 1
        model = DetrForObjectDetection(config)
        
        if os.path.exists(file_path):
            state = torch.load(file_path, map_location='cpu')
            model.load_state_dict(state)
            print(f"[DETR] Loaded model from {file_path}")
    
    model.to(device)
    
    data_loader = torch.utils.data.DataLoader(
        datasrc,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_DETR_skip_empty
    )
    
    # Optimizer & scheduler
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)
    
    # For long training (50+ epochs), use a more gradual learning rate schedule
    # Option 1: StepLR with multiple drops
    if epochs >= 100:
        # Drop LR every 1/4 of total epochs for very long training
        step_size = max(epochs // 4, 25)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size,
            gamma=0.1
        )
        print(f"[DETR] Using StepLR: dropping LR by 0.1x every {step_size} epochs")
    else:
        # For shorter training, drop at 1/3 point
        step_size = max(epochs // 3, 5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size,
            gamma=0.1
        )
        print(f"[DETR] Using StepLR: dropping LR by 0.1x every {step_size} epochs")
    
    # Option 2 (alternative): Cosine annealing - uncomment to use instead
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     T_max=epochs,
    #     eta_min=lr * 0.01
    # )
    # print(f"[DETR] Using CosineAnnealingLR from {lr} to {lr*0.01}")
    
    if trainAgain:
        model.train()
        
        # Training diagnostic (first batch only)
        first_batch_checked = False
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (images, targets) in enumerate(data_loader):
                imgs = list(images)
                annotations = [
                    {"image_id": i, "annotations": t["annotations"]} 
                    for i, t in enumerate(targets)
                ]
                
                # Process inputs
                encoding = processor(
                    images=imgs, 
                    annotations=annotations, 
                    return_tensors="pt", 
                    do_rescale=True
                )
                
                pixel_values = encoding["pixel_values"].to(device)
                labels = encoding["labels"]
                
                # Move labels to device
                hf_labels = [
                    {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                     for k, v in lab.items()} 
                    for lab in labels
                ]
                
                # Diagnostic for first batch
                if epoch == 0 and batch_idx == 0 and not first_batch_checked:
                    print("\n" + "="*70)
                    print("TRAINING DIAGNOSTIC (First Batch)")
                    print("="*70)
                    print(f"Model config: num_labels={model.config.num_labels}")
                    print(f"Number of images: {len(imgs)}")
                    print(f"Pixel values shape: {pixel_values.shape}")
                    print(f"Number of targets: {len(hf_labels)}")
                    if len(hf_labels) > 0:
                        label = hf_labels[0]
                        print(f"First target keys: {label.keys()}")
                        if 'boxes' in label:
                            print(f"  Boxes shape: {label['boxes'].shape}")
                            print(f"  First 3 boxes: {label['boxes'][:3].tolist()}")
                        if 'class_labels' in label:
                            print(f"  Class labels: {label['class_labels'].tolist()}")
                            print(f"  Unique classes: {label['class_labels'].unique().tolist()}")
                    print("="*70 + "\n")
                    first_batch_checked = True
                
                # Forward pass
                outputs = model(pixel_values=pixel_values, labels=hf_labels)
                loss = outputs.loss if outputs.loss is not None else sum(outputs.loss_dict.values())
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            lr_scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)
            
            print(f"[DETR] Epoch {epoch+1}/{epochs} - avg loss: {avg_loss:.6f}")
            if hasattr(outputs, 'loss_dict'):
                loss_dict_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in outputs.loss_dict.items()])
                print(f"  Loss components: {loss_dict_str}")
        
        # Save model
        torch.save(model.state_dict(), file_path)
        print(f"[DETR] Training finished, saved to {file_path}")
    else:
        model.eval()
        print(f"[DETR] Using existing model from {file_path}")
    
    return model



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
        batch_size = 1,
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

            # Don’t load COCO weights if you’ll be restoring your own checkpoint
            model = torchvision.models.detection.fcos_resnet50_fpn(weights=None)

            # Rebuild classification head with the right number of classes
            num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head = FCOSClassificationHead(
                in_channels=256,
                num_anchors=num_anchors,
                num_classes=num_classes,
                norm_layer=partial(torch.nn.GroupNorm, 32)
            )
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

        # memory cleanup
        #del images
        #del targets
        #del loss_dict
       #del losses
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
