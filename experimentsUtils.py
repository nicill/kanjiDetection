import configparser
import sys
import os
import time
import cv2
import torch

from pathlib import Path
from itertools import product

from datasets import ODDataset,ODDETRDataset

from config import read_config
from predict import detectBlobsMSER,detectBlobsDOG
from imageUtils import boxesFound,read_Binary_Mask,recoupMasks
from train import train_YOLO,makeTrainYAML, get_transform, train_pytorchModel,train_DETR, train_DeformableDETR

from dataHandlding import buildTRVT,buildNewDataTesting,separateTrainTest, forPytorchFromYOLO, buildTestingFromSingleFolderSakuma2, buildTestingFromSingleFolderSakuma2NOGT
from predict import predict_yolo, predict_pytorch, predict_DETR, predict_DeformableDETR_FIXED

from transformers import DetrForObjectDetection, DetrImageProcessor, DeformableDetrImageProcessor



class ModelExperiment:
    """Base class for running model experiments with single parameter sets"""
    
    def __init__(self, conf, device):
        self.conf = conf
        self.device = device
        self.results_file = conf["outTEXT"][:-4] + "SUMMARY" + conf["outTEXT"][-4:]
    
    def prepare_data(self):
        """Prepare training, validation, and test datasets"""
        if not self.conf["Prep"]:
            return
            
        buildTRVT(
            self.conf["Train_input_dir_images"], 
            self.conf["Train_input_dir_masks"],
            self.conf["slice"],
            os.path.join(self.conf["TV_dir"], self.conf["Train_dir"]), 
            os.path.join(self.conf["TV_dir"], self.conf["Valid_dir"]),
            os.path.join(self.conf["TV_dir"], self.conf["Test_dir"]),  
            self.conf["Train_Perc"], 
            doTest=False
        )
        
        buildTestingFromSingleFolderSakuma2(
            self.conf["Test_input_dir"],
            os.path.join(self.conf["TV_dir"], self.conf["Test_dir"]),
            self.conf["slice"],
            denoised=True
        )
    
    def write_results(self, params, metrics, train_time, test_time):
        """Write experiment results to file"""
        with open(self.results_file, "a+") as f:
            for v in params.values():
                f.write(f"{v},")
            f.write(f"{metrics['prec']},{metrics['rec']},{metrics['oprec']},{metrics['orec']},")
            f.write(f"{train_time},{test_time}\n")
    
    def train_and_test(self, params):
        """Train and test model - to be implemented by subclasses"""
        raise NotImplementedError


class YOLOExperiment(ModelExperiment):
    """YOLO model experiment"""
    
    def train_and_test(self, params):
        """Train and test YOLO with given parameters"""
        yaml_file = "trainEXP.yaml"
        prefix = "exp" + paramsDictToString(params)
        makeTrainYAML(self.conf, yaml_file, params)
        
        # Training
        train_time = 0
        if self.conf["Train"]:
            start = time.time()
            train_YOLO(self.conf, yaml_file, prefix, params=params)
            train_time = time.time() - start
        
        # Testing
        print("Testing YOLO model...")
        start = time.time()
        prec, rec, oprec, orec = predict_yolo(
            self.conf, 
            prefix + "epochs" + str(self.conf["ep"]) + 'ex'
        )
        test_time = time.time() - start
        
        metrics = {'prec': prec, 'rec': rec, 'oprec': oprec, 'orec': orec}
        return metrics, train_time, test_time


class PyTorchModelExperiment(ModelExperiment):
    """PyTorch models (Faster R-CNN, Mask R-CNN, etc.) experiment"""
    
    def __init__(self, conf, device):
        super().__init__(conf, device)
        self.num_classes = 2
        self.proportion = conf["Train_Perc"] / 100
        self.dataset = None
        self.dataset_test = None
    
    def load_datasets(self):
        """Load training and test datasets"""
        train_dir = os.path.join(self.conf["TV_dir"], self.conf["Train_dir"])
        test_dir = os.path.join(self.conf["TV_dir"], self.conf["Test_dir"])
        
        self.dataset = ODDataset(train_dir, True, self.conf["slice"], get_transform())
        self.dataset_test = ODDataset(test_dir, True, self.conf["slice"], get_transform())
        print(f"Train dataset length: {len(self.dataset)}")
    
    def train_and_test(self, params):
        """Train and test PyTorch model with given parameters"""
        if self.dataset is None:
            self.load_datasets()
        
        file_path = (
            "exp" + paramsDictToString(params, forFileName=True) + 
            "Epochs" + str(self.conf["ep"]) + ".pth"
        )
        print(f"Testing params {params} with file {file_path}")
        
        train_again = not Path(file_path).is_file()
        
        # Training
        train_time = 0
        if self.conf["Train"] or train_again:
            start = time.time()
            model = train_pytorchModel(
                dataset=self.dataset,
                device=self.device,
                num_classes=self.num_classes,
                file_path=file_path,
                num_epochs=self.conf["ep"],
                trainAgain=train_again,
                proportion=self.proportion,
                mType=params["modelType"],
                trainParams=params
            )
            train_time = time.time() - start
        
        # Testing
        start = time.time()
        pred_folder = os.path.join(self.conf["Pred_dir"], "exp" + paramsDictToString(params))
        orig_folder = os.path.join(self.conf["TV_dir"], self.conf["Test_dir"], "images")
        
        prec, rec, oprec, orec = predict_pytorch(
            dataset_test=self.dataset_test,
            model=model,
            device=self.device,
            predConfidence=params["predconf"],
            postProcess=0,
            predFolder=pred_folder,
            origFolder=orig_folder
        )
        test_time = time.time() - start
        
        metrics = {'prec': prec, 'rec': rec, 'oprec': oprec, 'orec': orec}
        return metrics, train_time, test_time


class DETRExperiment(ModelExperiment):
    """DETR model experiment"""
    
    def train_and_test(self, params):
        """Train and test DETR with given parameters"""
        file_path = (
            "DETR_exp" + paramsDictToString(params, forFileName=True) + 
            "Epochs" + str(self.conf["ep"]) + ".pth"
        )
        print(f"[DETR] Testing params {params} with file {file_path}")
        
        train_again = not Path(file_path).is_file()
        
        # Training
        train_time = 0
        if self.conf["Train"] or train_again:
            start = time.time()
            train_dir = os.path.join(self.conf["TV_dir"], self.conf["Train_dir"])
            detr_dataset = ODDETRDataset(train_dir, True, self.conf["slice"], get_transform())
            
            train_params = {
                "file_path": file_path,
                "trainAgain": train_again,
                "num_epochs": self.conf["ep"],
                "batch_size": params["batch_size"],
                "lr": params["lr"],
                "device": self.device
            }
            
            if params["modelType"] == "DETR":
                model = train_DETR(self.conf, detr_dataset, "DETR_exp_", train_params, file_path)
            elif params["modelType"] == "DEFDETR":
                model = train_DeformableDETR(self.conf, detr_dataset, "DETR_exp_", train_params, file_path)
            else:
                raise ValueError(f"Unknown DETR model type: {params['modelType']}")
            
            train_time = time.time() - start
        else:
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            state = torch.load(file_path, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device)
        
        # Testing
        start = time.time()
        test_dir = os.path.join(self.conf["TV_dir"], self.conf["Test_dir"])
        test_dataset = ODDETRDataset(test_dir, True, self.conf["slice"], get_transform())
        
        pred_folder = os.path.join(self.conf["Pred_dir"], "DETR_exp" + paramsDictToString(params))
        orig_folder = os.path.join(self.conf["TV_dir"], self.conf["Test_dir"], "images")
        
        if params["modelType"] == "DETR":
            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            prec, rec, oprec, orec = predict_DETR(
                test_dataset, model, processor, self.device,
                params["predconf"], params["max_detections"], params["resize"],
                pred_folder, orig_folder
            )
        elif params["modelType"] == "DEFDETR":
            processor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr")
            prec, rec, oprec, orec = predict_DeformableDETR_FIXED(
                test_dataset, model, processor, self.device,
                params["predconf"], params["max_detections"],
                pred_folder, orig_folder
            )
        
        test_time = time.time() - start
        
        metrics = {'prec': prec, 'rec': rec, 'oprec': oprec, 'orec': orec}
        return metrics, train_time, test_time


def DLExperiment(conf, yolo_params=None, pytorch_params=None, detr_params=None):
    """
    Run object detection experiments with single parameter sets
    
    Args:
        conf: Configuration dictionary
        yolo_params: Single dict of YOLO parameters (e.g., {"scale": 0.3, "mosaic": 0.5})
        pytorch_params: Single dict of PyTorch model parameters
        detr_params: Single dict of DETR parameters
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data preparation (only once)
    print("Preparing datasets...")
    base_experiment = ModelExperiment(conf, device)
    base_experiment.prepare_data()
    

    sys.exit()

    # Run YOLO experiment
    if yolo_params:
        print(f"\n=== Running YOLO Experiment ===")
        print(f"Parameters: {yolo_params}")
        experiment = YOLOExperiment(conf, device)
        try:
            metrics, train_time, test_time = experiment.train_and_test(yolo_params)
            experiment.write_results(yolo_params, metrics, train_time, test_time)
            print(f"YOLO Results: Precision={metrics['prec']:.3f}, Recall={metrics['rec']:.3f}")
        except Exception as e:
            print(f"YOLO experiment failed: {e}")
    
    # Run PyTorch model experiment
    if pytorch_params:
        print(f"\n=== Running PyTorch Model Experiment ===")
        print(f"Parameters: {pytorch_params}")
        experiment = PyTorchModelExperiment(conf, device)
        try:
            metrics, train_time, test_time = experiment.train_and_test(pytorch_params)
            experiment.write_results(pytorch_params, metrics, train_time, test_time)
            print(f"PyTorch Results: Precision={metrics['prec']:.3f}, Recall={metrics['rec']:.3f}")
        except Exception as e:
            print(f"PyTorch experiment failed: {e}")
    
    # Run DETR experiment
    if detr_params:
        print(f"\n=== Running DETR Experiment ===")
        print(f"Parameters: {detr_params}")
        experiment = DETRExperiment(conf, device)
        try:
            metrics, train_time, test_time = experiment.train_and_test(detr_params)
            experiment.write_results(detr_params, metrics, train_time, test_time)
            print(f"DETR Results: Precision={metrics['prec']:.3f}, Recall={metrics['rec']:.3f}")
        except Exception as e:
            print(f"DETR experiment failed: {e}")
    
    print("\n=== All Experiments Complete ===")


# Example usage:
if __name__ == "__main__":
    print("IS CUDA AVAILABLE???????????????????????")
    print(torch.cuda.is_available())

    doYolo = True
    doPytorch = False
    doDETR = False

    # Configuration file name, can be entered in the command line
    configFile = "config.ini" if len(sys.argv) < 2 else sys.argv[1]

    # DL experiment
    conf = read_config(configFile)
    print(conf)
    
    # Define single parameter sets
    yolo_params = {"scale": 0.3, "mosaic": 0.5} if doYolo else None
    pytorch_params = {"modelType": "fasterrcnn", "score": 0.25, "nms": 0.5, "predconf": 0.7} if doPytorch else None
    detr_params = {"modelType": "DETR", "lr": 5e-6, "batch_size": 8, "predconf": 0.5, 
                   "nms_iou": 0.5, "max_detections": 50, "resize": 800} if doDETR else None
    
    # Run experiments
    DLExperiment(conf, yolo_params, pytorch_params, detr_params)