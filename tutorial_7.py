import crypten
import torch
import subprocess
crypten.init()
torch.set_num_threads(1)

subprocess.run(['python3', './mnist_utils.py', '--option', 'features', '--reduced', '100', '--binary'])

import torch.nn as nn
import torch.nn.functional as F


# Define an example network
class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(16 * 12 * 12, 100)
        self.fc2 = nn.Linear(100, 2)  # For binary classification, final layer needs only 2 outputs

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 16 * 12 * 12)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


crypten.common.serial.register_safe_class(ExampleNet)


# Define source argument values for Alice and Bob
ALICE = 0
BOB = 1

# Load Alice's data
data_alice_enc = crypten.load_from_party('/tmp/alice_train.pth', src=ALICE)

# We'll now set up the data for our small example below
# For illustration purposes, we will create toy data
# and encrypt all of it from source ALICE
x_small = torch.rand(100, 1, 28, 28)
y_small = torch.randint(1, (100,))

# Transform labels into one-hot encoding
label_eye = torch.eye(2)
y_one_hot = label_eye[y_small]

# Transform all data to CrypTensors
x_train = crypten.cryptensor(x_small, src=ALICE)
y_train = crypten.cryptensor(y_one_hot)

# Instantiate and encrypt a CrypTen model
model_plaintext = ExampleNet()
dummy_input = torch.empty(1, 1, 28, 28)
model = crypten.nn.from_pytorch(model_plaintext, dummy_input)
model.encrypt()

# Example: Stochastic Gradient Descent in CrypTen

model.train()  # Change to training mode
loss = crypten.nn.MSELoss()  # Choose loss functions

# Set parameters: learning rate, num_epochs
learning_rate = 0.001
num_epochs = 2

# Train the model: SGD on encrypted data
for i in range(num_epochs):
    # forward pass
    output = model(x_train)
    loss_value = loss(output, y_train)

    # set gradients to zero
    model.zero_grad()

    # perform backward pass
    loss_value.backward()

    # update parameters
    model.update_parameters(learning_rate)

    # examine the loss after each epoch
    print("Epoch: {0:d} Loss: {1:.4f}".format(i, loss_value.get_plain_text()))

import crypten.mpc as mpc
import crypten.communicator as comm

# Convert labels to one-hot encoding
# Since labels are public in this use case, we will simply use them from loaded torch tensors
labels = torch.load('/tmp/train_labels.pth')
labels = labels.long()
labels_one_hot = label_eye[labels]


@mpc.run_multiprocess(world_size=2)
def run_encrypted_training():
    # Load data:
    x_alice_enc = crypten.load_from_party('/tmp/alice_train.pth', src=ALICE)
    x_bob_enc = crypten.load_from_party('/tmp/bob_train.pth', src=BOB)

    crypten.print(x_alice_enc.size())
    crypten.print(x_bob_enc.size())

    # Combine the feature sets: identical to Tutorial 3
    x_combined_enc = crypten.cat([x_alice_enc, x_bob_enc], dim=2)

    # Reshape to match the network architecture
    x_combined_enc = x_combined_enc.unsqueeze(1)

run_encrypted_training()

import os

filenames = ['/tmp/alice_train.pth',
             '/tmp/bob_train.pth',
             '/tmp/alice_test.pth',
             '/tmp/bob_test.pth',
             '/tmp/train_labels.pth',
             '/tmp/test_labels.pth']

for fn in filenames:
    if os.path.exists(fn): os.remove(fn)