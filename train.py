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

from torchvision.transforms.functional import to_pil_image

from utils import MetricLogger,SmoothedValue,warmup_lr_scheduler,reduce_dict,collate_fn
from imageUtils import boxListEvaluation
import math
import time
import cv2
import numpy as np

from PIL import Image

torch.backends.cudnn.benchmark = True


def makeTrainYAML(conf, fileName = 'trainAUTO.yaml'):
    """
    Function to write a yaml file
    """
    data = { "names": {0: 'Kanji'}, "path": conf["TV_dir"], "train": conf["Train_dir"],
            "val": conf["Valid_dir"] }
    with open(fileName, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def train_YOLO(conf, datasrc):
    """
    Train a YOlO model from a
    train and validation folder
    """
    resfolder = conf["Train_res"]
    epochs = conf["ep"]
    name = 'combined_data_'+str(epochs)+'ex'
    valfolder = conf["Valid_res"]
    imgsize = conf["slice"]
    resultstxt = os.path.join(resfolder,valfolder,'results_' + name + '.txt')

    settings.update({'runs_dir':resfolder})
    model = YOLO('yolov8n.pt')
    results = model.train(data=datasrc,epochs=epochs,
                imgsz=imgsize,name=name,device=0)
    results = model.val(project=resfolder,name=valfolder,save_json=True)

    with open(resultstxt,'w+') as res:
        res.write(str(results))


def train_pytorchModel(folder, num_epochs = 10, proportion = 0.9):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print("CUDA AVAILABLE")
    else:
        raise Exception("Train pytorch model, CUDA is not available, raising exception")

    # save model
    file_path = "fasterrcnn_resnet50_fpn.pth"
    trainAgain = True
    
    # our dataset has two classes only - background and person
    num_classes = 2
    bs = 16
    # use our dataset and defined transformations
    dataset = ODDataset(sys.argv[1],1500, get_transform(train=True))
    dataset_test = ODDataset(sys.argv[1],1500, get_transform(train=True))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    divide = int(len(dataset)*proportion)
    dataset = torch.utils.data.Subset(dataset, indices[:divide])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[divide:])

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    # probably have a look at this  https://discuss.pytorch.org/t/how-to-use-collate-fn/27181/2
    print("Training Dataset Length "+str(len(dataset)))
    print("Testing Dataset Length "+str(len(dataset_test)))

    if trainAgain :
        # get the model using our helper function
        model = get_model_instance_segmentation(num_classes)
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
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)  # No pretrained weights
        #this shoudl probably be in get_model_instance_segmentation

        # Replace the box predictor with one matching the saved model's class count (2 classes, including background)
        num_classes = 2  # Adjust to match the number of classes in your saved model
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

        model.to(device)

        # Load the saved state_dict into the model
        model.load_state_dict(torch.load(file_path))
        model.eval()  # Set the model to evaluation mode



    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

    print("That's it!")

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

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    # to change the model, see, for example
    # https://github.com/pytorch/tutorials/blob/d686b662932a380a58b7683425faa00c06bcf502/intermediate_source/torchvision_tutorial.rst

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    print("evaluating "+str(len(data_loader)))

    # store all images in disk (debugging purposes)
    count = 0
    for images,targets in data_loader:
        for im in images:
            to_pil_image(im).save("./debug/testIM"+str(count)+".png")
            count+=1

    count=0
    for images, targets in data_loader:
        # store test images to disk
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        #print(targets[0]["boxes"])
        #print(outputs[0]["boxes"])

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        masks = outputs[0]["masks"]
        #binary_masks = (masks > 0.5).to(torch.uint8)

        masks = (masks > 0.5).squeeze(1).cpu().numpy()  # Threshold and convert to NumPy
        accumMask = masks[0]*255
        for i, mask in enumerate(masks):
            accumMask[mask>0]=255

        outMask = Image.fromarray((255-accumMask).astype("uint8"), mode="L")
        outMask.save("./debug/TestIM"+str(count)+"mask.png")
        #cv2.imwrite("./debug/TestIM"+str(count)+"mask.jpg", (255-accumMask) )
        """
        accumMask = np.moveaxis(masks[0].numpy(),0,-1)
        print("image "+str(count)+" has masks "+str(len(masks)))
        for m in masks:
            m2 = np.moveaxis(m.numpy(),0,-1)
            accumMask[m2>0] = 255
        cv2.imwrite("./debug/TestIM"+str(count)+"mask.jpg",255-accumMask)
        """

        evaluator_time = time.time()
        print("Evaluating ")
        print(boxListEvaluation(outputs[0]["boxes"],targets[0]["boxes"]))
        evaluator_time = time.time() - evaluator_time
        print("time "+str(evaluator_time))
        count+=1

    # accumulate predictions from all images
    torch.set_num_threads(n_threads)

if __name__ == '__main__':
    num_epochs = 5
    train_pytorchModel(sys.argv[1],num_epochs)
