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


from dataHandlding import buildTRVT,buildNewDataTesting,separateTrainTest
from train import train_YOLO,makeTrainYAML,get_transform, train_pytorchModel
from predict import predict_yolo, predict_pytorch

from datasets import ODDataset

def read_config(filename):
    conf = configparser.ConfigParser()
    conf.read(filename)
    # print config parser
    #print({section: dict(conf[section]) for section in conf.sections()})
    res_dict = {}

    section = 'TODO'
    res_dict["Prep"] = conf[section].get('preprocess') == "yes"
    res_dict["Train"] = conf[section].get('train') == "yes"
    res_dict["Test"] = conf[section].get('test') == "yes"
    res_dict["Post"] = conf[section].get('postprocess') == "yes"
    res_dict["DLN"] = conf[section].get('network')

    section = 'PREP'
    res_dict["Train_input_dir_images"] = conf[section].get('trainSourceImages')
    res_dict["Train_input_dir_masks"] = conf[section].get('trainSourceMasks')
    res_dict["createTest"] = conf[section].get('doTestFolder') == "yes"
    res_dict["torchData"]= conf[section].get('pytorchDataFolder')

    section = 'TRAIN'
    res_dict["slice"] = int(conf[section].get('sliceSize'))

    res_dict["TV_dir"] = conf[section].get('tVDir')
    res_dict["Train_dir"] = conf[section].get('trainDir')
    res_dict["Valid_dir"] = conf[section].get('validDir')
    res_dict["Test_dir"] = conf[section].get('testDir')
    res_dict["Train_Perc"] = int(conf[section].get('trainPercentage'))

    res_dict["Train_res"] = conf[section].get('trainResFolder')
    res_dict["Valid_res"] = conf[section].get('valResFolder')
    res_dict["ep"] = int(conf[section].get('epochs'))
    res_dict["again"] = conf[section].get('trainagain') == "yes"

    res_dict["pScoreTH"] = float(conf[section].get('pScoreTH'))
    res_dict["pnmsTH"] = float(conf[section].get('pnmsTH'))

    section = 'TEST'
    res_dict["Test_ND_dir"] = conf[section].get('testNewDataDir')
    res_dict["models"] = conf[section].get('modelist').strip().split(",")
    res_dict["pmodel"] = conf[section].get('pmodel')

    res_dict["Pred_dir"] = conf[section].get('predDir')
    res_dict["Exp_dir"] = conf[section].get('expDir')
    res_dict["Masks_dir"] = conf[section].get('newMasksDir')

    return res_dict

def main(fName):
    """
    main function, read command line parameters
    and call the adequate functions
    """
    conf = read_config(fName)
    print(conf)

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
            # use the GPU or the CPU, if a GPU is not available
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            if torch.cuda.is_available():
                print("CUDA AVAILABLE")
            else:
                raise Exception("Train pytorch model, CUDA is not available, raising exception")

            # our dataset has two classes only - background and Kanji
            num_classes = 2
            bs = 16 # should probably be a parameter
            proportion = conf["Train_Perc"]/100
            # change so it reads the train dataset from one folder
            # and the test from another
            # use our dataset and defined transformations
            separateTrainTest(conf["torchData"],os.path.join(conf["torchData"],"separated"),proportion)
            dataset = ODDataset(os.path.join(conf["torchData"],"separated","train"),conf["slice"], get_transform())
            dataset_test = ODDataset(os.path.join(conf["torchData"],"separated","test"),conf["slice"], get_transform())

            """
            dataset = ODDataset(conf["torchData"],conf["slice"], get_transform())
            indices = torch.randperm(len(dataset)).tolist()
            divide = int(len(dataset)*proportion)
            dataset = torch.utils.data.Subset(dataset, indices[:divide])
            dataset_test = ODDataset(conf["torchData"],conf["slice"], get_transform())
            dataset_test = torch.utils.data.Subset(dataset_test, indices[divide:])
            """

    if conf["Train"]:
        print("train!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if conf["DLN"] == "YOLO":
            yamlTrainFile = "trainAUTO.yaml"
            makeTrainYAML(conf,yamlTrainFile)
            train_YOLO(conf,yamlTrainFile)
        elif conf["DLN"] == "FRCNN":
            tParams = {"score":conf["pScoreTH"],"nms":conf["pnmsTH"]}
            # there is a proportion parameter that we may or may not want to touch
            pmodel = train_pytorchModel(dataset = dataset, device = device,
            num_classes = num_classes, file_path = conf["pmodel"],
            num_epochs = conf["ep"], trainAgain=conf["again"],
            proportion = proportion, trainParams = tParams)

    if conf["Test"]:
        print("test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if conf["DLN"] == "YOLO":
            predict_yolo(conf)
        elif conf["DLN"] == "FRCNN":
            # parameter that controls the confidence of the boxes
            predConf = 0.7
            predict_pytorch(dataset_test = dataset_test, model = pmodel,
            device = device, predConfidence = predConf)

if __name__ == "__main__":

    # Configuration file name, can be entered in the command line
    configFile = "config.ini" if len(sys.argv) < 2 else sys.argv[1]
    main(configFile)
