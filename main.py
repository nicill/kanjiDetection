"""
   File to call the control the flow
   of an application to detect Kanji
   In Wasan documents
   using Deep Learning networks
"""

import configparser
import sys
import yaml

from buildTrainValidation import buildTrainValid
from train import train_YOLO

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
    if res_dict["Prep"] :
        res_dict["Train_input_dir_images"] = conf[section].get('trainSourceImages')
        res_dict["Train_input_dir_masks"] = conf[section].get('trainSourceMasks')

    section = 'TRAIN'
    if res_dict["Prep"] or res_dict["Train"]:
        res_dict["slice"] = int(conf[section].get('sliceSize'))

        res_dict["TV_dir"] = conf[section].get('tVDir')
        res_dict["Train_dir"] = conf[section].get('trainDir')
        res_dict["Valid_dir"] = conf[section].get('validDir')
        res_dict["Train_Perc"] = int(conf[section].get('trainPercentage'))

        res_dict["Train_res"] = conf[section].get('trainResFolder')
        res_dict["Valid_res"] = conf[section].get('valResFolder')
        res_dict["ep"] = int(conf[section].get('epochs'))

    return res_dict

def makeTrainYAML(conf, fileName = 'trainAUTO.yaml'):
    """
    Function to write a yaml file
    """
    data = { "names": {0: 'Kanji'}, "path": conf["TV_dir"], "train": conf["Train_dir"],
            "val": conf["Valid_dir"] }
    with open(fileName, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def main(fName):
    """
    main function, read command line parameters
    and call the adequate functions
    """
    conf = read_config(fName)
    print(conf)

    # Do whatever needs to be done
    if conf["Prep"]:
        buildTrainValid(conf["Train_input_dir_images"],
        conf["Train_input_dir_masks"],conf["slice"],conf["Train_dir"],
        conf["Valid_dir"],conf["Train_Perc"])

    if conf["Train"]:
        yamlTrainFile = "trainAUTO.yaml"
        makeTrainYAML(conf,yamlTrainFile)
        train_YOLO(conf,yamlTrainFile)

if __name__ == "__main__":

    # Configuration file name, can be entered in the command line
    configFile = "config.ini" if len(sys.argv) < 2 else sys.argv[1]
    main(configFile)
