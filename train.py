import os
from ultralytics import YOLO
from ultralytics import settings
import torch
torch.backends.cudnn.benchmark = True

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
