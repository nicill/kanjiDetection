"""
   File to call the control the flow
   of an application to detect Kanji
   In Wasan documents
   using Deep Learning networks
"""

import configparser
import sys
import os
import torch


from dataHandlding import buildTRVT,buildNewDataTesting,separateTrainTest, forPytorchFromYOLO
from train import train_YOLO,makeTrainYAML,get_transform, train_pytorchModel
from predict import predict_yolo, predict_new_Set_yolo, predict_pytorch

from datasets import ODDataset
from config import read_config

def main(fName):
    """
    main function, read command line parameters
    and call the adequate functions
    """
    conf = read_config(fName)
    print(conf)

    # use the GPU or the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Do whatever needs to be done
    if conf["Prep"]:
        if conf["DLN"] == "YOLO":
            print("calling btrvt")
            buildTRVT(conf["Train_input_dir_images"],
            conf["Train_input_dir_masks"],conf["slice"],
            os.path.join(conf["TV_dir"],conf["Train_dir"]),
            os.path.join(conf["TV_dir"],conf["Valid_dir"]),
            os.path.join(conf["TV_dir"],conf["Test_dir"]),
            conf["Train_Perc"])

            if ["createTest"]:
                print("create test")
                buildNewDataTesting(conf["Train_input_dir_images"],
                conf["Train_input_dir_masks"], conf["Test_dir"])
        elif conf["DLN"] == "FRCNN":

            # change so it reads the train dataset from one folder
            # and the test from another
            # use our dataset and defined transformations
            if False: separateTrainTest(conf["torchData"],os.path.join(conf["torchData"],"separated"),proportion)
            else:
                forPytorchFromYOLO(os.path.join(conf["TV_dir"],conf["Train_dir"]),
                os.path.join(conf["TV_dir"],conf["Valid_dir"]),
                os.path.join(conf["TV_dir"],conf["Test_dir"]), 
                conf["torchData"] )

    if conf["Train"]:
        print("train!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if conf["DLN"] == "YOLO":
            yamlTrainFile = "trainAUTO.yaml"
            makeTrainYAML(conf,yamlTrainFile)
            train_YOLO(conf,yamlTrainFile)
        elif conf["DLN"] == "FRCNN":
            if torch.cuda.is_available():
                print("CUDA AVAILABLE")
            else:
                raise Exception("Train pytorch model, CUDA is not available, raising exception")

            # our dataset has two classes only - background and Kanji
            num_classes = 2
            bs = 32 # should probably be a parameter
            proportion = conf["Train_Perc"]/100 
            # parameter in case we want to separate or use the one created by YOLO

            yoloFormat = conf["yoloFormat"]
            if yoloFormat:
                dataset = ODDataset(os.path.join(conf["torchData"],"train"), yoloFormat, conf["slice"], get_transform())
                dataset_test = ODDataset(os.path.join(conf["torchData"],"test"), yoloFormat, conf["slice"], get_transform())
            else:
                dataset = ODDataset(os.path.join(conf["torchData"],"separated","train"), yoloFormat, conf["slice"], get_transform())
                dataset_test = ODDataset(os.path.join(conf["torchData"],"separated","test"), yoloFormat, conf["slice"], get_transform())


            tParams = {"score":conf["pScoreTH"],"nms":conf["pnmsTH"]}
            # there is a proportion parameter that we may or may not want to touch
            pmodel = train_pytorchModel(dataset = dataset, device = device,
            num_classes = num_classes, file_path = conf["pmodel"],
            num_epochs = conf["ep"], trainAgain=conf["again"],
            proportion = proportion, trainParams = tParams)

    if conf["Test"]:
        print("test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        newSet = False
        if not newSet:
            if conf["DLN"] == "YOLO":
                predict_yolo(conf)
            elif conf["DLN"] == "FRCNN":
                # parameter that controls the confidence of the boxes
                predConf = 0.7
                predict_pytorch(dataset_test = dataset_test, model = pmodel,
                device = device, predConfidence = predConf)
        else:
            predict_new_Set_yolo(conf)

if __name__ == "__main__":

    # Configuration file name, can be entered in the command line
    configFile = "config.ini" if len(sys.argv) < 2 else sys.argv[1]
    main(configFile)
