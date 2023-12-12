# Imports here
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import os

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type = str, help = 'path to the directory of dataset')
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'directory to save checkpoint(s)')
parser.add_argument('--arch', type = str, default = 'vgg19', help='choose architecture vgg19, alexnet, or densenet121')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'choose learning rate for training')
parser.add_argument('--hidden_units', type = int, default = 256, help = 'choose number of hidden units' )
parser.add_argument('--epochs', type = int, default = 30, help = 'set number of epochs')
parser.add_argument('--gpu', action = 'store_true', help = 'Use GPU or not')

args = parser.parse_args()

# Directories
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# Transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

data_transforms = {'train': train_transforms, 'valid': valid_transforms}

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])

image_datasets = {'train': train_data, 'valid': valid_data}

# dataloaders using the image datasets and the trainforms
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

dataloaders = {'train': trainloader, 'valid': validloader}

# Checking for GPU
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

#Loading pre-trained model
model = models.vgg19(pretrained=True)

#Parameter freeze
for param in model.parameters():
    param.requires_grad = False

#Defining new classifier

input_units = {'vgg19': 25088, 'alexnet': 9216, 'densenet121': 1024}

classifier = nn.Sequential(nn.Linear(input_units[args.arch], args.hidden_units),
                           nn.ReLU(),
                           nn.Dropout(0.3),
                           nn.Linear(args.hidden_units, 102),
                           nn.LogSoftmax(dim=1))

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
model.to(device)

# Training the network

epochs = args.epochs

for e in range(epochs):
    running_loss = 0
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    else:
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                valid_loss += batch_loss.item()
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        model.train()

        print(f"Epoch {e + 1}/{epochs}.. "
              f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
              f"Validaion loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
              f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

# Defining and saving checkpoint

checkpoint = {'classifier': classifier,
              'class_to_idx': image_datasets['train'].class_to_idx,
              'n_epochs': epochs,
              'optimizer_state': optimizer.state_dict(),
              'model_state': model.state_dict(),
              'arch': args.arch}

torch.save(checkpoint, args.save_dir)