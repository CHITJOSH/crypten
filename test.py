import crypten
import torch
import crypten.nn as nn
import crypten.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# Set up CrypTen
crypten.init()

# Define the CNN model
#class CNN(nn.Module):
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

crypten.common.serial.register_safe_class(CNN)
