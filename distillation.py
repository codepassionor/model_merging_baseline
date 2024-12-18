import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from torchsummary import summary
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torch import nn, optim

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# Data Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform2 = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize to Inception v3's expected input size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize for CIFAR-10
])



# Load CIFAR10 Dataset
# Update paths to a writable location
trainset = datasets.CIFAR10('./train', download=True, train=True, transform=transform)
valset = datasets.CIFAR10('./val', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False)

len_trainset = len(trainset)
len_valset = len(valset)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Device Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ResNet50 Teacher Model
print('--------------------loading resnet50 model!--------------------')
resnet = models.resnet50(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False

num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)
resnet = resnet.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.fc.parameters())

# Training Function
def train_and_evaluate(model, trainloader, valloader, criterion, optimizer, len_trainset, len_valset, num_epochs=25):
    model.train()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len_trainset
        epoch_acc = running_corrects.double() / len_trainset
        print(f' Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        model.eval()
        running_loss_val = 0.0
        running_corrects_val = 0

        for inputs, labels in valloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_loss_val += loss.item() * inputs.size(0)
                running_corrects_val += torch.sum(preds == labels.data)

        epoch_loss_val = running_loss_val / len_valset
        epoch_acc_val = running_corrects_val.double() / len_valset

        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f' Val Loss: {epoch_loss_val:.4f} Acc: {epoch_acc_val:.4f}')
        print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

# # Train the ResNet Teacher Model
# print('--------------------training the resnet teacher!!--------------------')
# resnet_teacher = train_and_evaluate(resnet, trainloader, valloader, criterion, optimizer, len_trainset, len_valset, 10)
# # Student Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.pool1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Knowledge Distillation Loss
def loss_kd(outputs, labels, teacher_outputs, temperature, alpha):
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / temperature, dim=1),
                             F.softmax(teacher_outputs / temperature, dim=1)) * (alpha * temperature ** 2)
    KD_loss += F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

# Get Teacher Outputs
def get_outputs(model, dataloader):
    outputs = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs_batch = model(inputs).cpu().numpy()
            outputs.append(outputs_batch)
    return outputs

# Train and Evaluate KD Model
def train_and_evaluate_kd(model, teacher_model, optimizer, loss_kd, trainloader, valloader, temperature, alpha, num_epochs=25):
    teacher_model.eval()
    best_model_wts = copy.deepcopy(model.state_dict())
    outputs_teacher_train = get_outputs(teacher_model, trainloader)
    outputs_teacher_val = get_outputs(teacher_model, valloader)

    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Training the student network
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for i, (images, labels) in enumerate(trainloader):
            inputs = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            teacher_out = torch.from_numpy(outputs_teacher_train[i]).to(device)
            loss = loss_kd(outputs, labels, teacher_out, temperature, alpha)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_acc = running_corrects.double() / len_trainset
        print(f' Train Loss: {running_loss / len_trainset:.4f} Acc: {epoch_acc:.4f}')

    return model

# Initialize and Train Student Model
net = Net().to(device)
print('--------------------training the resnet50 student model!!--------------------')
stud = train_and_evaluate_kd(net, resnet_teacher, optim.Adam(net.parameters()), loss_kd,
                             trainloader, valloader, temperature=1, alpha=0.5, num_epochs=20)
save_model(stud, "resnet_student_net.pth")
print('--------------------saved the resnet_student model!--------------------')
print('--------------------loading vgg16 model!--------------------')
vgg_16 = models.vgg16(pretrained=True)

# Freeze all layers
for param in vgg_16.parameters():
    param.requires_grad = False

vgg_16.classifier[6] = nn.Linear(vgg_16.classifier[6].in_features, 10)

# Move the model to the device
vgg_16 = vgg_16.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg_16.classifier[6].parameters())

print('--------------------training the vgg_16 teacher!!--------------------')
vgg_teacher = train_and_evaluate(vgg_16, trainloader, valloader, criterion, optimizer, len_trainset, len_valset, 10)


print('--------------------training the vgg_16 student model!!--------------------')
net = Net().to(device)
print('--------------------training the resnet50 student model!!--------------------')
stud = train_and_evaluate_kd(net, vgg_teacher, optim.Adam(net.parameters()), loss_kd,
                             trainloader, valloader, temperature=1, alpha=0.5, num_epochs=20)
save_model(stud, "vgg_student_net.pth")
print('--------------------saved the vgg_student model!--------------------')




print('--------------------loading mobilenet_v2 model!--------------------')
mobilenet_v2 = models.mobilenet_v2(pretrained=True)

# Freeze all layers
for param in mobilenet_v2.parameters():
    param.requires_grad = False

mobilenet_v2.classifier[1] = nn.Linear(mobilenet_v2.last_channel, 10)

# Move the model to the device
mobilenet_v2 = mobilenet_v2.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mobilenet_v2.classifier[1].parameters())

print('--------------------training the mobilenet_v2 teacher!!--------------------')
mobilenet_teacher = train_and_evaluate(mobilenet_v2, trainloader, valloader, criterion, optimizer, len_trainset, len_valset, 10)

print('--------------------training the mobilenet_v2 student model!!--------------------')
net = Net().to(device)
stud = train_and_evaluate_kd(net, mobilenet_teacher, optim.Adam(net.parameters()), loss_kd,
                             trainloader, valloader, temperature=1, alpha=0.5, num_epochs=20)
save_model(stud, "mobilenet_student_net.pth")
print('--------------------saved the mobilenet_student model!--------------------')





print('--------------------loading densenet121 model!--------------------')
densenet121 = models.densenet121(pretrained=True)

# Freeze all layers
for param in densenet121.parameters():
    param.requires_grad = False

densenet121.classifier = nn.Linear(densenet121.classifier.in_features, 10)

# Move the model to the device
densenet121 = densenet121.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(densenet121.classifier.parameters())

print('--------------------training the densenet121 teacher!!--------------------')
densenet_teacher = train_and_evaluate(densenet121, trainloader, valloader, criterion, optimizer, len_trainset, len_valset, 10)

print('--------------------training the densenet121 student model!!--------------------')
net = Net().to(device)
print('--------------------training the resnet50 student model!!--------------------')
stud = train_and_evaluate_kd(net, densenet_teacher, optim.Adam(net.parameters()), loss_kd,
                             trainloader, valloader, temperature=1, alpha=0.5, num_epochs=20)
save_model(stud, "densenet_student_net.pth")
print('--------------------saved the densenet_student model!--------------------')




print('--------------------loading inception_v3 model!--------------------')
inception_v3 = models.inception_v3(pretrained=True)

# Freeze all layers
for param in inception_v3.parameters():
    param.requires_grad = False

trainset = datasets.CIFAR10('./train', download=True, train=True, transform=transform2)
valset = datasets.CIFAR10('./val', download=True, train=False, transform=transform2)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False)

len_trainset = len(trainset)
len_valset = len(valset)

inception_v3.fc = nn.Linear(inception_v3.fc.in_features, 10)

# Move the model to the device
inception_v3 = inception_v3.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(inception_v3.fc.parameters())

print('--------------------training the inception_v3 teacher!!--------------------')
inception_teacher = train_and_evaluate(inception_v3, trainloader, valloader, criterion, optimizer, len_trainset, len_valset, 10)

print('--------------------training the inception_v3 student model!!--------------------')
net = Net().to(device)
stud = train_and_evaluate_kd(net, inception_teacher, optim.Adam(net.parameters()), loss_kd,
                             trainloader, valloader, temperature=1, alpha=0.5, num_epochs=20)
save_model(stud, "inception_student_net.pth")
print('--------------------saved the inception_student model!--------------------')
dense_model = models.densenet121(pretrained=False)
print(dense_model)