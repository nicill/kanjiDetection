import os
from ultralytics import YOLO
from ultralytics import settings
import torch
import yaml

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
