# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import os
import copy
import time
import sys
import numpy as np
from collections import defaultdict
from datasets import CPDataset,tDataset
from imageUtils import read_Binary_Mask


def testAndOutputForAnnotations(inFolder,outFileName,model,weights, classDict):
    """
    After a model has previously been trained,
    and Kanji images have been detected
    read some detected images from a folder
    classify them and output the three higher probability
    classes into a file. Receive a class dictionary
    to translate between class codes and unicode
    """
    # the model and class dict come as parameters
    # make a list of all kanji images and another of their names
    def firstRow(num):
        """
        create a simple
        string for the first
        row in the file
        """
        ret="imageName"
        for i in range(num):
            ret+=","+"Pred"+str(i+1)
        return ret+"\n"

    def toUnicode(s):
        """
        transform a class code into
        a printable unicode character
        """
        return chr(int(s[2:],16))
    def formatOut(aList):
        """
        Simple function to return a list of
        values in pretty text format
        """
        retString = ""
        for x in aList[::-1]:
            retString += ","+toUnicode(classDict[x])
        return retString
    bs = 64
    showPreds = 5
    model.eval()

    imNames = []
    ims = []
    for dirpath, dnames, fnames in os.walk(inFolder):
        for f in fnames:
            # ignore "context" images
            if "CONTEXT" not in f:
                ims.append(read_Binary_Mask(os.path.join(inFolder,f)))
                imNames.append(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create Dataset and dataloader  so that it can be fed to the GPU
    dummyDS = tDataset(ims,weights.transforms())
    dummyDL = torch.utils.data.DataLoader(dummyDS, batch_size=bs, shuffle=False)

    totalPreds = []
    with torch.no_grad():
        for inputs,targets in dummyDL:
            if torch.cuda.is_available():
                inputs = inputs.to(device)
                outputs = model(inputs)
                #_, preds = torch.max(outputs, 1)
                preds = torch.argsort(outputs, 1)
                totalPreds.extend(preds.to('cpu').detach().numpy().copy())
    #print(totalPreds)
    # now write the predictions to file
    with open(outFileName, mode='w', encoding="utf-8") as f:
        # write first line
        f.write(firstRow(showPreds))
        for i in range(len(imNames)):
            f.write(imNames[i]+formatOut(totalPreds[i][-showPreds:])+"\n")

def train(model, criterion, optimizer, dataloader, sizes, device):

    model.train() # set model to training mode
    running_loss = 0.0
    running_corrects = 0
    #print("lenght of the dataloader before loop "+str(len(dataloader)))
    data_count=0
    for inputs, target in dataloader:
        #print("inside the FOR LOOP AT CNN_RESNET50 "+str(count))
        if torch.cuda.is_available():
            inputs, target = inputs.to(device), target.to(device)
            data_count+=len(inputs)
            #print('cuda is available')

            optimizer.zero_grad() # zero the parameter gradients

            # forward pass
            with torch.set_grad_enabled(True):
                #print("inside the forward pass")
                outputs = model(inputs)

                _, preds = torch.max(outputs, 1)

                # loss function
                # print("output shape is "+str(outputs))
                # print("target shape is "+str(target))
                loss = criterion(outputs, target)

                # back-propagation
                loss.backward()

                # optimization calculations
                optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == target.data)

        else:
            raise Exception("cuda is not available !!!!!!")
            # print("cuda is not available !!!!!!")

    #epoch loss and accuracy
    epoch_loss = running_loss / sizes
    epoch_acc = running_corrects / sizes
    epoch_acc=epoch_acc.to('cpu').detach().numpy().copy()
    return epoch_loss, epoch_acc

def validation(model, criterion, dataloader, sizes, device):
    # set model to evaluate mode
    model.eval()

    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0
        data_count=0
        for inputs, target in dataloader:
            if torch.cuda.is_available():
                inputs, target = inputs.to(device), target.to(device)
                data_count+=len(inputs)

                outputs = model(inputs)

                _, preds = torch.max(outputs, 1)


                # loss function
                loss = criterion(outputs, target)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == target.data)

            else:
                raise Exception("cuda is not available !!!!!!")

    epoch_loss = running_loss / sizes
    epoch_acc = running_corrects / sizes
    epoch_acc = epoch_acc.to('cpu').detach().numpy().copy()
    return epoch_loss, epoch_acc

def duplicate_rename(file_path):
    if os.path.exists(file_path):
        name, ext = os.path.splitext(file_path)
        i = 1
        while True:
            new_name = '{} ({}){}'.format(name, i, ext)
            if not os.path.exists(new_name):
                return new_name
            i += 1
    else:
        return file_path

def train_model(model, criterion, optimizer, dataloaderTrain, dataloaderTest, sizeTrain, sizeTest, device, num_epochs=25, save_model=True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = defaultdict(list)

    print('training start !!!!!')
    print("-"*64)

    for epoch in range(num_epochs):
        string = f'| epoch {epoch + 1}/{num_epochs}:'+" "*50+"|"
        print(string)

        train_loss, train_acc = train(model, criterion, optimizer,  dataloaderTrain, sizeTrain, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        val_loss, val_acc = validation(model, criterion, dataloaderTest, sizeTest, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print("| train loss:  {:.4f} | train acc:  {:.4f} | val acc:  \033[31m{:.4f}\033[0m  |".format(train_loss, train_acc, val_acc))
        else:
            print("| train loss:  {:.4f} | train acc:  {:.4f} | val acc:  {:.4f}  |".format(train_loss, train_acc,val_acc))
        # print(f'| [train] Loss: {train_loss:.6f}, Acc: {train_acc:.6f}')
        # print(f'| [test] Loss: {val_loss:.6f}, Acc: {val_acc:.6f}')
        print('-'*64)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:6f}'.format(best_acc))

    # save best model weights
    pth = f'best_wts_{model.__class__.__name__}50_epo'+str(num_epochs)+'.pth'
    new_pth = duplicate_rename(pth)

    if save_model:
        torch.save(best_model_wts, new_pth)
        print("model parameters was saved...")

    # print(history)
    return best_acc, history

def plot_loss_acc(history, model, num_epochs):
    epochs = np.arange(1, num_epochs + 1)
    _, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))
    # transition of loss
    ax1.set_title("Loss")
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    # transition of accuracy
    ax2.set_title("Acc")
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.savefig(f'result_{model.__class__.__name__}.jpg')

def loadModelReadClassDict(arch,modelFile,cDfile):
    """
    Reads a class dictionary from a problem
    as well as a model file given its architecture
    """
    # first, read class dictionary
    with open(cDfile) as f:
        classDict = { int(line.strip().split(",")[0]):line.strip().split(",")[1] for line in f.readlines() }
    outputNumClasses =len(classDict)
    #print(outputNumClasses)

    if arch == "resnet":
        weights = models.ResNet50_Weights.DEFAULT
        model_ft = models.resnet50(weights=weights)
    elif arch == "swimt":
        weights = models.Swin_T_Weights.DEFAULT
        model_ft = models.swin_t(weights=weights)
    elif arch == "swims":
        weights = models.Swin_S_Weights.DEFAULT
        model_ft = models.swin_s(weights=weights)
    elif arch == "swimb":
        weights = models.Swin_B_Weights.DEFAULT
        model_ft = models.swin_b(weights=weights)
        # swim B needs smaller batch size because of memory requirements
        bs = 16
    elif arch == "vitb16":
        weights = models.ViT_B_16_Weights.DEFAULT
        model_ft = models.vit_b_16(weights=weights)
    elif arch == "vitb32":
        weights = models.ViT_B_32_Weights.DEFAULT
        model_ft = models.vit_b_32(weights=weights)
    elif arch == "vith14":
        weights = models.ViT_H_14_Weights.DEFAULT
        model_ft = models.vit_h_14(weights=weights)
    elif arch == "vitl16":
        weights = models.ViT_L_16_Weights.DEFAULT
        model_ft = models.vit_l_16(weights=weights)
    elif arch == "vitl32":
        weights = models.ViT_L_32_Weights.DEFAULT
        model_ft = models.vit_l_32(weights=weights)
    elif arch == "convnexts":
        weights = models.convnext.ConvNeXt_Small_Weights.DEFAULT
        model_ft = models.convnext_small(weights=weights)
    elif arch == "convnextt":
        weights = models.convnext.ConvNeXt_Tiny_Weights.DEFAULT
        model_ft = models.convnext_tiny(weights=weights)
    elif arch == "convnextb":
        weights = models.convnext.ConvNeXt_Base_Weights.DEFAULT
        model_ft = models.convnext_base(weights=weights)
        bs = 16
    elif arch == "convnextl":
        weights = models.convnext.ConvNeXt_Large_Weights.DEFAULT
        model_ft = models.convnext_large(weights=weights)
        bs = 8
    else: raise Exception("Unrecognized architecture")

    # Adapt the model to our number of classes, depending on the architecture
    if arch == "resnet":
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, outputNumClasses)
    elif arch == "swimt" or arch == "swims" or arch == "swimb" :
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, outputNumClasses)
    elif arch == "vitb16" or arch == "vitb32" or arch == "vith14" or arch == "vitl16" or arch == "vitl32" :
        num_ftrs = model_ft.heads[0].in_features
        model_ft.heads[0] = nn.Linear(num_ftrs, outputNumClasses)
    elif arch == "convnexts" or arch =="convnextt" or arch =="convnextb" or arch =="convnextl":
        num_ftrs = model_ft.classifier[-1].in_features
        model_ft.classifier[-1] = nn.Linear(num_ftrs, outputNumClasses)

    #load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_ft = model_ft.to(device)

    model_ft.load_state_dict(torch.load(modelFile, map_location=device))
    return model_ft,weights,classDict

def main(argv):

    arch = "resnet"
    bs = 256 #default batch size

    if len(argv)>2: arch = argv[2]

    #model definition
    if arch == "resnet":
        """ OLD DEFINITION
        model_ft = models.resnet50(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, outputNumClasses)
        """
        weights = models.ResNet50_Weights.DEFAULT
        model_ft = models.resnet50(weights=weights)
    elif arch == "swimt":
        weights = models.Swin_T_Weights.DEFAULT
        model_ft = models.swin_t(weights=weights)
    elif arch == "swims":
        weights = models.Swin_S_Weights.DEFAULT
        model_ft = models.swin_s(weights=weights)
    elif arch == "swimb":
        weights = models.Swin_B_Weights.DEFAULT
        model_ft = models.swin_b(weights=weights)
        # swim B needs smaller batch size because of memory requirements
        bs = 16
    elif arch == "vitb16":
        weights = models.ViT_B_16_Weights.DEFAULT
        model_ft = models.vit_b_16(weights=weights)
    elif arch == "vitb32":
        weights = models.ViT_B_32_Weights.DEFAULT
        model_ft = models.vit_b_32(weights=weights)
    elif arch == "vith14":
        weights = models.ViT_H_14_Weights.DEFAULT
        model_ft = models.vit_h_14(weights=weights)
    elif arch == "vitl16":
        weights = models.ViT_L_16_Weights.DEFAULT
        model_ft = models.vit_l_16(weights=weights)
    elif arch == "vitl32":
        weights = models.ViT_L_32_Weights.DEFAULT
        model_ft = models.vit_l_32(weights=weights)
    elif arch == "convnexts":
        weights = models.convnext.ConvNeXt_Small_Weights.DEFAULT
        model_ft = models.convnext_small(weights=weights)
    elif arch == "convnextt":
        weights = models.convnext.ConvNeXt_Tiny_Weights.DEFAULT
        model_ft = models.convnext_tiny(weights=weights)
    elif arch == "convnextb":
        weights = models.convnext.ConvNeXt_Base_Weights.DEFAULT
        model_ft = models.convnext_base(weights=weights)
        bs = 16
    elif arch == "convnextl":
        weights = models.convnext.ConvNeXt_Large_Weights.DEFAULT
        model_ft = models.convnext_large(weights=weights)
        bs = 8
    else: raise Exception("Unrecognized architecture "+str(arch))

    print("using architecture "+arch)

    # data transforms now depend on model weights
    data_transforms = {'train': weights.transforms(), 'val': weights.transforms()}

    # input train data folder
    data_dir = argv[1]

    # create data !!
    dataset=CPDataset(data_dir,data_transforms["train"])
    #dataset=CPDataset(data_dir)

    # split into train data and test data
    #n_val=int(len(dataset) * 0.2)
    #n_train=len(dataset) - n_val
    #print("the number of train dataset is "+str(n_train)+", the number of validation dataset is "+str(n_val))
    #train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    proportion = 0.1
    train_dataset, val_dataset = dataset.breakTrainValid(proportion)
    print("the size of the train dataset is "+str(len(train_dataset))+", the number of validation dataset is "+str(len(val_dataset)))

    # create dataloader
    dataloaders_train = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    dataloaders_val = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=True)

    dataset_train_sizes = len(train_dataset)
    dataset_val_sizes = len(val_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("my device is "+str(device))

    outputNumClasses = dataset.numClasses()

    # Adapt the model to our number of classes, depending on the architecture
    if arch == "resnet":
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, outputNumClasses)
    elif arch == "swimt" or arch == "swims" or arch == "swimb" :
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, outputNumClasses)
    elif arch == "vitb16" or arch == "vitb32" or arch == "vith14" or arch == "vitl16" or arch == "vitl32" :
        num_ftrs = model_ft.heads[0].in_features
        model_ft.heads[0] = nn.Linear(num_ftrs, outputNumClasses)
    elif arch == "convnexts" or arch =="convnextt" or arch =="convnextb" or arch =="convnextl":
        num_ftrs = model_ft.classifier[-1].in_features
        model_ft.classifier[-1] = nn.Linear(num_ftrs, outputNumClasses)


    print(f"the number of out put classes is {outputNumClasses}")

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adamax(model_ft.parameters(), lr=0.0002)

    # set epoch sizes
    epo = 10

    # ResNet50
    _,history = train_model(
                        model_ft, criterion, optimizer_ft, dataloaders_train, dataloaders_val,
                        dataset_train_sizes, dataset_val_sizes, device, epo, True
                        )
    plot_loss_acc(history, model_ft, epo)

if __name__ == '__main__':
    #mod,w,cD = loadModelReadClassDict(sys.argv[1], sys.argv[2], sys.argv[3])
    #testAndOutputForAnnotations(sys.argv[4],sys.argv[5],mod,w,cD)
    main(sys.argv)
