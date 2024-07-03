"""
   File to call the control the flow
   of an application to detect Kanji
   In Wasan documents
   using Deep Learning networks
"""

import configparser
import sys
import os

from buildTrainValidation import buildTrainValid,buildTesting
from train import train_YOLO,makeTrainYAML
from predict import predict_yolo

def read_config(filename):
    conf = configparser.ConfigParser()
    conf.read(filename)
    res_dict = {}

    section = 'TODO'
    res_dict["Prep"] = conf[section].get('preprocess') == "yes"
    res_dict["Train"] = conf[section].get('train') == "yes"
    res_dict["Test"] = conf[section].get('test') == "yes"
    res_dict["Post"] = conf[section].get('postprocess') == "yes"

    section = 'PREP'
    res_dict["Train_input_dir_images"] = conf[section].get('trainSourceImages')
    res_dict["Train_input_dir_masks"] = conf[section].get('trainSourceMasks')
    res_dict["createTest"] = conf[section].get('doTestFolder') == "yes"

    section = 'TRAIN'
    res_dict["slice"] = int(conf[section].get('sliceSize'))

    res_dict["TV_dir"] = conf[section].get('tVDir')
    res_dict["Train_dir"] = conf[section].get('trainDir')
    res_dict["Valid_dir"] = conf[section].get('validDir')
    res_dict["Train_Perc"] = int(conf[section].get('trainPercentage'))

    res_dict["Train_res"] = conf[section].get('trainResFolder')
    res_dict["Valid_res"] = conf[section].get('valResFolder')
    res_dict["ep"] = int(conf[section].get('epochs'))

    section = 'TEST'
    res_dict["Test_dir"] = conf[section].get('testDir')
    res_dict["models"] = conf[section].get('modelist').strip().split(",")
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
        print("calling btv")
        buildTrainValid(conf["Train_input_dir_images"],
        conf["Train_input_dir_masks"],conf["slice"],os.path.join(conf["TV_dir"],conf["Train_dir"]),
        os.path.join(conf["TV_dir"],conf["Valid_dir"]),conf["Train_Perc"])

        if ["createTest"]:
            print("create test")
            buildTesting(conf["Train_input_dir_images"],
            conf["Train_input_dir_masks"], conf["Test_dir"])

    if conf["Train"]:
        print("train!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        yamlTrainFile = "trainAUTO.yaml"
        makeTrainYAML(conf,yamlTrainFile)
        train_YOLO(conf,yamlTrainFile)

    if conf["Test"]:
        predict_yolo(conf)

if __name__ == "__main__":

    # Configuration file name, can be entered in the command line
    configFile = "config.ini" if len(sys.argv) < 2 else sys.argv[1]
    main(configFile)
