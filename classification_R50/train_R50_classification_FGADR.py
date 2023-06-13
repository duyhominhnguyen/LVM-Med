import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.datasets as datasets
import argparse
from tqdm import tqdm
import os
from utils.func import (
    parse_config,
    load_config
)
# Define dataset class
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['0', '1', '2', '3', '4']
        self.class_to_idx = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
        self.images = []
        self.labels = []

        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

new_settings = {
    "resnet50": {
        "ssl-2d-3d" : "/home/caduser/KOTORI/jsrt-downstream/ssl-2d-3d-weight/converted_vissl_pretrained.torch",
        "ssl-graph-match-small" : "/home/caduser/KOTORI/jsrt-downstream/ssl-2d-3d-weight/converted_vissl_ssl_graph_small.torch",
            "imagenet"   : "/home/caduser/KOTORI/WEIGHTS/Supervised/resnet50-19c8e357.pth",
        "uni-jigsaw" : "/home/caduser/KOTORI/WEIGHTS/PALIWAL_pretrained/checkpoints_jigsaw_120epochs_4V100_16GB/model_final_checkpoint_phase119.torch",
        "uni-simclr" : "/home/caduser/KOTORI/WEIGHTS/PALIWAL_pretrained/checkpoints_simclr_100epochs_4A100/model_final_checkpoint_phase99.torch",
        "uni-swav"   : "/home/caduser/KOTORI/WEIGHTS/PALIWAL_pretrained/checkpoints_swav_100epochs_4A100/model_final_checkpoint_phase99.torch",
        "uni-rotnet" : "/home/caduser/KOTORI/WEIGHTS/PALIWAL_pretrained/checkpoints_rotnet_120epochs_4A100/model_final_checkpoint_phase119.torch",
        "uni-dino"   : "/home/caduser/KOTORI/WEIGHTS/PALIWAL_pretrained/checkpoints_dino_100epochs_4A100/model_final_checkpoint_phase99.torch",
        "uni-moco"   : "/home/caduser/KOTORI/WEIGHTS/PALIWAL_pretrained/checkpoints_moco_180epochs_4A100/model_final_checkpoint_phase179.torch",
        "uni-twin"   : "/home/caduser/KOTORI/WEIGHTS/PALIWAL_pretrained/checkpoints_barlow_100epochs_4A100/model_final_checkpoint_phase99.torch",
        "uni-deepc"  : "/home/caduser/KOTORI/WEIGHTS/PALIWAL_pretrained/checkpoints_deepcluster_100epochs_4A100_16GCPU/model_final_checkpoint_phase99.torch",
        "usst-swav-2D"   : "/home/caduser/KOTORI/WEIGHTS/USST/usst_swav_2D_100epochs/model_final_checkpoint_phase99.torch",
        "usst-swav-all"  : "/home/caduser/KOTORI/WEIGHTS/USST/usst_swav_all_200epochs/model_final_checkpoint_phase199.torch",
        "usst-deepc-2D"  : "/home/caduser/KOTORI/WEIGHTS/USST/usst_deepc_2D_100epochs/model_final_checkpoint_phase99.torch",
        "usst-deepc-all" : "/home/caduser/KOTORI/WEIGHTS/USST/usst_deepc_all_200epochs/model_final_checkpoint_phase199.torch",
        "vissl-1M-twin"   : "/home/caduser/KOTORI/WEIGHTS/VISSL_pretrained/twin/model_final_checkpoint_phase199.pth",
        "vissl-1M-deepc"  : "/home/caduser/KOTORI/WEIGHTS/VISSL_pretrained/deepc/model_iteration936510.torch",
        "vissl-1M-dino"   : "/home/caduser/KOTORI/WEIGHTS/VISSL_pretrained/dino/model_final_checkpoint_phase199.pth",
        "vissl-1M-simclr" : "/home/caduser/KOTORI/WEIGHTS/VISSL_pretrained/simclr/model_final_checkpoint_phase199.torch",
        "vissl-1M-moco"   : "/home/caduser/KOTORI/WEIGHTS/VISSL_pretrained/moco/model_final_checkpoint_phase199.torch",
        "vissl-1M-swav"   : "/home/caduser/KOTORI/WEIGHTS/VISSL_pretrained/swav/model_final_checkpoint_phase199.torch",
        "vissl-1M-graph-match": "/home/caduser/KOTORI/WEIGHTS/VISSL_pretrained/graph_match/converted_vissl_ssl_graph_large.torch",
        "vissl-1M-vicrel" : "/home/caduser/KOTORI/WEIGHTS/VISSL_pretrained/vicregl/model_100epochs.pth",
        "vissl-imnet-deepc" : "/home/caduser/KOTORI/WEIGHTS/ImageNet/deepclusterv2_400ep_pretrain.pth.tar",
        "vissl-imnet-swav"  : "/home/caduser/KOTORI/WEIGHTS/ImageNet/swav_rn50_200ep.torch",
        "vissl-imnet-simclr": "/home/caduser/KOTORI/WEIGHTS/ImageNet/simclr_rn50_200ep.torch",
        "vissl-imnet-moco"  : "/home/caduser/KOTORI/WEIGHTS/ImageNet/moco_rn50_200ep.torch",
        "vissl-imnet-twin"  : "/home/caduser/KOTORI/WEIGHTS/ImageNet/twins_rn50_300ep.torch",
        "vissl-imnet-dino"  : "/home/caduser/KOTORI/WEIGHTS/ImageNet/dino_resnet50_pretrain.pth",
        "clip-resnet50"  : "/home/caduser/KOTORI/downstream_task/CLIP_pretrained/RN50_vision_model.pt"
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
    # val_loss /= len(val_indices)
    val_loss /= num_samples
    val_acc /= num_samples
    net.train()

    return val_loss, val_acc

def TrainingTesting(cfg, numtry, pretrained_weight_name, device, solver, name_weights, linear_eval, number_epoch = 50, learning_rate = 0.001, batch_size = 32):
# Define transforms for data augmentation
    train_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        #transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets
    train_dir = cfg.dataloader.train_dir_img
    test_dir = cfg.dataloader.test_dir_img

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
  
    # Split the training dataset into training and validation sets

    train_indices, val_indices = train_test_split(list(range(len(train_dataset))), test_size=0.2, random_state=42)
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
    if linear_eval:
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)
    # path2weights="./models/resnet18_pretrained.pt"
    # model.load_state_dict(torch.load(path2weights)
    pretrained_weight = torch.load(new_settings["resnet50"][pretrained_weight_name], map_location=device)
    model.load_state_dict(pretrained_weight, strict=False)
    print("Loaded pretrained-weight of ", pretrained_weight_name)
    # if torch.cuda.is_available():
    #      device = torch.device("cuda")

    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    print ("Using solver " + solver)
    if solver == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    if solver == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

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
                #print(inputs.size())
                #print(labels.size())
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
            torch.save(checkpoint, cfg.base.best_valid_model_checkpoint
                       + name_weights + "_" + pretrained_weight_name+str(numtry)+".pth")
            print("Saved checkpoint at epoch ", epoch + 1)
            best_acc_val = val_acc

        print(f"Training Loss: {train_loss:.4f}\t Training Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}\tVal Accuracy: {val_acc:.5f}")

## ------------ Test the model ------------
    print("------ Testing ------")
    ckp = torch.load(best_valid_model_checkpoint
                     + name_weights + "_" + pretrained_weight_name+str(numtry)+".pth")
    model.load_state_dict(ckp['state_dict'])
    num_samples_test = len(test_dataset)
    test_loss, test_acc = eval(model, test_loader, device, criterion, num_samples_test)
    print(f"Test Loss: {test_loss:.4f}\tTest Accuracy: {test_acc:.5f}")
    return test_acc



if __name__ == "__main__":
    yml_args = parse_config()
    cfg = load_config(yml_args.config)


    list_acc = []

    # args.pretrained = "vissl-1M-swav"
    # name_weight = args.pretrained + "_v1"

    pretrained = cfg.base.original_checkpoint
    name_weight = pretrained + "_v1"

    epochs = cfg.train.num_epochs
    lr = cfg.train.learning_rate
    train_batch_size = cfg.train.train_batch_size
    GPUs = cfg.base.gpu_id
    solver_name = cfg.train.optimizer
    linear_eval = cfg.train.linear_eval
    # solver_name = 'sgd'

    cuda_string = 'cuda:' + GPUs
    devices = torch.device(cuda_string if torch.cuda.is_available() else 'cpu')

    for numtry in range(3):
        print("Trial ", numtry)
        test_acc = TrainingTesting(cfg = cfg, numtry=numtry, pretrained_weight_name=pretrained, device= devices, solver= solver_name,
                                   name_weights=name_weight, linear_eval = linear_eval,
                                   number_epoch=epochs,
                        learning_rate=lr, batch_size=train_batch_size)
        list_acc.append(test_acc.to('cpu'))
        print("==============================================================================")

    print("Mean Accuracy: ", np.mean(list_acc))
    print("Standard Deviation: ", np.std(list_acc))

