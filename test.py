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


def test_FLOWERS(model, test_loader):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f'Training in {device}')
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        model.eval()
        running_loss = 0
        running_error = 0
        correct = 0
        total = 0

        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            running_error += (predicted != labels).long().sum().item()
            correct += (predicted == labels).long().sum().item()
            total += labels.size(0)

        avg_test_error = running_error/len(test_loader.dataset)
        avg_test_loss = running_error/len(test_loader)
        test_acc = correct/total

        t = time.time() - start_time
        print(f'Test_Loss = {avg_test_loss:.4f}, Test_Error = {avg_test_error:.4f}, Test_Acc = {test_acc:.4%}')
        # print(f'Time after Epoch {epoch+1}: {t:2f}s')
        model_path = model_name(model.name, batch_size, lr, epoch+1)

        # torch.save(model.state_dict(), model_path)