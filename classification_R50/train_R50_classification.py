"""
Linear and non-linear image classification tasks with and w/o frozen image encoders
"""
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.datasets as datasets
import argparse
from tqdm import tqdm
import os
import random
import torch.nn.functional as F
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



weight_collections = {
    "resnet50": {
        "lvm-med-resnet": "./lvm_med_weights/lvmmed_resnet.torch",
    }
}

def eval(net, dataloader, device, criterion, num_samples):
    # Evaluate the model on the validation set
    val_loss = 0.0
    val_acc = 0.0

    net.eval()
    with tqdm(total=len(dataloader), desc='Validation round', unit=' img') as pbar:
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_acc += torch.sum(preds == labels.data)
            pbar.update(inputs.shape[0])
    val_loss /= num_samples
    val_acc /= num_samples
    net.train()
    return val_loss, val_acc

def TrainingTesting(cfg, numtry, pretrained_weight_name, data_path, num_classes, data_tranform, device, solver, name_weights, 
                    frozen_encoder, architecture_type, number_epoch=50, learning_rate=0.001, batch_size=32, test_mode='best_valid', 
                    valid_rate=0.2):
    # Load the datasets
    train_dir = data_path + "/Training"
    test_dir = data_path + "/Testing"
    if frozen_encoder:
        checkpoint_dir = cfg.base.best_valid_model_checkpoint + cfg.base.dataset_name + "_" + architecture_type + "_" + name_weights + "_frozen/"
    else:
        checkpoint_dir = cfg.base.best_valid_model_checkpoint + cfg.base.dataset_name + "_" + architecture_type + "_" + name_weights + "_non_frozen/"

    CHECK_FOLDER = os.path.isdir(checkpoint_dir)

    if not CHECK_FOLDER:
        os.makedirs(checkpoint_dir)
        print("created folder: ", checkpoint_dir)

    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_tranform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=data_tranform)

    print ("valid size is {}".format(valid_rate))
    # Split the training dataset into training and validation sets
    train_indices, val_indices = train_test_split(list(range(len(train_dataset))), test_size=valid_rate, random_state=42)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    loader_args = dict(num_workers=10, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, **loader_args)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **loader_args)

    n_train = len(train_indices)

    # Define the ResNet50 model
    model = torchvision.models.resnet50(pretrained=True)

    # Freeze the layers of the ResNet50 model
    if frozen_encoder:
        print ("Frozen encoder")
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features

    # define linear or non-linear architecture
    if architecture_type == '1-fcn':
        print ("Using single fully-connected layer")
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif architecture_type == "fcns":
        print("Using several fully-connected layers")
        if cfg.base.dataset_name == 'brain':
            model.fc = nn.Sequential(
                                  nn.Linear(num_ftrs, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, num_classes))
        elif cfg.base.dataset_name == 'fgadr':
            model.fc = nn.Sequential(
                                  nn.Linear(num_ftrs, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, num_classes))
        else:
            print(">>> Not implemented for selected datasets")
            exit()
    else:
        print (">>> No available option for achitecture. Please check 'help' with --linear option")
        exit()

    pretrained_weight = torch.load(weight_collections["resnet50"][pretrained_weight_name], map_location=device)
    model.load_state_dict(pretrained_weight, strict=False)
    print("Loaded pretrained-weight of ", pretrained_weight_name)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    if solver == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif solver == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=0.)
        if cfg.base.dataset_name == 'fgadr':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=0.005)
    else:
        print ("Non-available solver")
        exit()

    ## ------------ Train the model ------------
    print(" ------------ Training ------------ ")
    num_epochs = number_epoch
    best_acc_val = 0.

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}:")
        train_loss = 0.0
        train_acc = 0.0

        # Train the model on the training set
        model.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                train_acc += torch.sum(preds == labels.data)

                # updating progressing bar
                pbar.update(inputs.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Print the results for this epoch
        train_loss /= len(train_indices)
        train_acc /= len(train_indices)
        num_samples = len(val_indices)
        num_samples_test = len(test_dataset)

        print(" \n >>> Evaluation ")
        val_loss, val_acc = eval(model, val_loader, device, criterion, num_samples)       

        if val_acc >= best_acc_val:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, checkpoint_dir
                       + name_weights + "_" + pretrained_weight_name + "_" + str(numtry) + ".pth")
            print("Saved checkpoint at epoch ", epoch + 1)
            best_acc_val = val_acc

        print(f"Training Loss: {train_loss:.4f}\t Training Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}\tVal Accuracy: {val_acc:.5f}")

    # print model at last epochs
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, checkpoint_dir
               + name_weights + "_" + pretrained_weight_name + "_last_" + str(numtry) + ".pth")
    print("Saved checkpoint at last epoch ", epoch + 1)

    ## ------------ Test the model ------------
    print("------ Testing ------")
    if test_mode == "best_valid":
        print("Loading best models at {}".format(checkpoint_dir))
        ckp = torch.load(checkpoint_dir
                         + name_weights + "_" + pretrained_weight_name + "_" + str(numtry) + ".pth")
    else:
        print("Loading models at last epochs {}".format(checkpoint_dir))
        ckp = torch.load(checkpoint_dir
                         + name_weights + "_" + pretrained_weight_name + "_last_" + str(numtry) + ".pth")
    model.load_state_dict(ckp['state_dict'])
    num_samples_test = len(test_dataset)
    test_loss, test_acc = eval(model, test_loader, device, criterion, num_samples_test)
    print(f"Test Loss: {test_loss:.4f}\tTest Accuracy: {test_acc:.5f}")
    return test_acc

def inference(numtry, device, cfg, data_path, data_tranform, name_weights, pretrained_weight_name,
              frozen_encoder, architecture_type, num_classes):
    if frozen_encoder:
        checkpoint_dir = cfg.base.best_valid_model_checkpoint + cfg.base.dataset_name + "_" + architecture_type + "_" + name_weights + "_frozen/"
    else:
        checkpoint_dir = cfg.base.best_valid_model_checkpoint + cfg.base.dataset_name + "_" + architecture_type + "_" + name_weights + "_non_frozen/"
    loader_args = dict(num_workers=10, pin_memory=True)
    test_dir = data_path + "/Testing"
    test_dataset = datasets.ImageFolder(root=test_dir, transform=data_tranform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, **loader_args)
    
    model = torchvision.models.resnet50(pretrained=True)
    # Freeze the layers of the ResNet50 model
    if frozen_encoder:
        print ("Frozen encoder")
        for param in model.parameters():
            param.requires_grad = False
            
    num_ftrs = model.fc.in_features   
    if architecture_type == '1-fcn':
        print ("Using single fully-connected layer")
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif architecture_type == "fcns":
        print("Using several fully-connected layers")
        if cfg.base.dataset_name == 'brain':
            model.fc = nn.Sequential(
                                  nn.Linear(num_ftrs, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, num_classes))
        elif cfg.base.dataset_name == 'fgadr':
            model.fc = nn.Sequential(
                                  nn.Linear(num_ftrs, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, num_classes))
        else:
            print(">>> Not implemented for selected datasets")
            exit()
    else:
        print (">>> No available option for achitecture. Please check 'help' with --linear option")
        exit()

    model = model.to(device)
    print("Loading best models at {}".format(checkpoint_dir))
    ckp = torch.load(checkpoint_dir
                         + name_weights + "_" + pretrained_weight_name + "_" + str(numtry) + ".pth")
    model.load_state_dict(ckp['state_dict'])
    num_samples_test = len(test_dataset)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = eval(model, test_loader, device, criterion, num_samples_test)
    print(f"Test Loss: {test_loss:.4f}\tTest Accuracy: {test_acc:.5f}")
    return test_acc

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_R50(yml_args, cfg):

    if cfg.base.dataset_name == 'brain':
        data_path = cfg.dataloader.data_path
        num_classes = 4
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif cfg.base.dataset_name == 'fgadr':
        data_path = cfg.dataloader.data_path
        num_classes = 5
        data_transforms = transforms.Compose([
            transforms.RandomCrop(size=(480, 480)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAutocontrast(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        print (">>> No available datasets")
        exit()

    print ("Using dataset {}".format(cfg.base.dataset_name))
    list_acc = []

    name_weight = cfg.base.original_checkpoint + "_output"
    cuda_string = 'cuda:' + cfg.base.gpu_id
    devices = torch.device(cuda_string if torch.cuda.is_available() else 'cpu')

    if not yml_args.use_test_mode:
        # Training model with three trial times
        for numtry in range(3):
            print ("*****"*3 + "\n" + "Trial", numtry)
            test_acc = TrainingTesting(cfg = cfg, numtry=numtry, pretrained_weight_name=cfg.base.original_checkpoint, data_path = data_path,
                                       num_classes = num_classes,
                                       data_tranform = data_transforms,
                                       device=devices,
                                       solver=cfg.train.solver,
                                       name_weights=name_weight, frozen_encoder=cfg.base.frozen_eval,
                                       number_epoch=cfg.train.num_epochs, architecture_type=cfg.base.model,
                                       learning_rate=cfg.train.learning_rate, batch_size=cfg.train.train_batch_size, 
                                       test_mode=cfg.base.test_mode,
                                       valid_rate = cfg.base.valid_rate)
            list_acc.append(test_acc.to('cpu'))
            print("==============================================================================")
        print ("*****"*3 + "\n")
        print("Mean Accuracy: ", np.mean(list_acc))
        print("Standard Deviation: ", np.std(list_acc))
    else:
        # Evaluate model with three weights
        for numtry in range(3):
            print ("*****"*3 + "\n" + "weight", numtry+1)
            test_acc = inference(numtry = numtry, device = devices, cfg = cfg, data_path = data_path, data_tranform=data_transforms,
                                 name_weights=name_weight, pretrained_weight_name=cfg.base.original_checkpoint,
                                 frozen_encoder=cfg.base.frozen_eval, architecture_type=cfg.base.model, num_classes=num_classes)
            list_acc.append(test_acc.to('cpu'))
        print ("*****"*3 + "\n")
        print("Mean Accuracy: ", np.mean(list_acc))
        print("Standard Deviation: ", np.std(list_acc))
