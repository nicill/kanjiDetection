"""
   File to call the control the flow
   of an application to detect Kanji
   In Wasan documents 
   using Deep Learning networks
"""

import configparser
import sys

def read_config(filename):
    conf = configparser.ConfigParser()
    conf.read(filename)
    res_dict = {}

    section = 'TODO'
    res_dict["Prep"] = conf[section].get('preprocess') == "yes"
    res_dict["Train"] = conf[section].get('train') == "yes"
    res_dict["Test"] = conf[section].get('test') == "yes"
    res_dict["Post"] = conf[section].get('postprocess') == "yes"

    section = 'TRAIN'
    if res_dict["Prep"] or res_dict["Train"]:
        res_dict["Train_input_dir_images"] = conf[section].get('trainSourceImages')
        res_dict["Train_input_dir_masks"] = conf[section].get('trainSourceMasks')
        res_dict["Train_dir"] = conf[section].get('trainDir')
        res_dict["Valid_dir"] = conf[section].get('validDir')
        res_dict["Train_Perc"] = conf[section].get('trainPercentage')

    return res_dict



def main(fName):
    """
    main function, read command line parameters
    and call the adequate functions
    """
    conf = read_config(fName)
    print(conf)


if __name__ == "__main__":

    # Configuration file name, can be entered in the command line
    configFile = "config.ini" if len(sys.argv) < 2 else sys.argv[1]
    main(configFile)