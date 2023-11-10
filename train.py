import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.datasets
import torchvision.transforms

import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import time


def train_FLOWERS(model, train_loader, val_loader, batch_size=128, lr=0.001, epoch=10):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f'Training in {device}')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epoch):
        running_loss = 0
        running_error = 0
        correct = 0
        total = 0

        model.train()

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() #
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() #
            optimizer.step() #

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            running_error += (predicted != labels).long().sum().item()
            correct += (predicted == labels).long().sum().item()
            total += labels.size(0)

        avg_train_error = running_error/len(train_loader.dataset)
        avg_train_loss = running_error/len(train_loader)
        train_acc = correct/total

        model.eval() #

        with torch.no_grad():
            running_loss = 0
            running_error = 0
            correct = 0
            total = 0

            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)

                running_error += (predicted != labels).long().sum().item()
                correct += (predicted == labels).long().sum().item()
                total += labels.size(0)

            avg_val_error = running_error/len(val_loader.dataset)
            avg_val_loss = running_error/len(val_loader)
            val_acc = correct/total

        t = time.time() - start_time
        print(f'Epoch {epoch+1}: Train_Loss = {avg_train_loss:.4f}, Train_Error = {avg_train_error:.4f}, Train_Acc = {train_acc:.4%} || Val_Loss = {avg_val_loss:.4f}, Val_Error = {avg_val_error:.4f}, Val_Acc = {val_acc:.4%}')
        print(f'Time after Epoch {epoch+1}: {t:2f}s')
        model_path = model_name(model.name, batch_size, lr, epoch+1)
        
        torch.save(model.state_dict(), model_path)

        # torch.save(model.state_dict(), model_path)