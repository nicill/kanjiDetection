"""
   File to manage the configuration of our different experiments
"""

import configparser

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

    res_dict["Test_input_dir"] = conf[section].get('testSource')

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
    res_dict["yoloFormat"] = conf[section].get('yoloFormat') == "yes"


    section = 'TEST'
    res_dict["Test_ND_dir"] = conf[section].get('testNewDataDir')
    res_dict["models"] = conf[section].get('modelist').strip().split(",")
    res_dict["pmodel"] = conf[section].get('pmodel')

    res_dict["Pred_dir"] = conf[section].get('predDir')
    res_dict["Exp_dir"] = conf[section].get('expDir')
    res_dict["Masks_dir"] = conf[section].get('newMasksDir')

    return res_dict
