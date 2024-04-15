import crypten
import torch

crypten.init()
torch.set_num_threads(1)

# Define ALICE and BOB src values
ALICE = 0
BOB = 1

# Ignore warnings
import warnings;
warnings.filterwarnings("ignore")

# Keep track of all created temporary files so that we can clean up at the end
temp_files = []

import torch.nn as nn
import os
# Instantiate single Linear layer
layer_linear = nn.Linear(4, 2)

# The weights and the bias are initialized to small random values
print("Plaintext Weights:\n\n", layer_linear._parameters['weight'])
print("\nPlaintext Bias:\n\n", layer_linear._parameters['bias'])

project_folder = os.path.dirname(os.path.abspath(__file__))
layer_linear_file = os.path.join(project_folder, "/tmp/tutorial5_layer_alice1.pth")
# Save the plaintext layer
crypten.save(layer_linear, layer_linear_file)
temp_files.append(layer_linear_file)

# Generate some toy data
features = 4
examples = 3
toy_data = torch.rand(examples, features)

# Save the plaintext toy data
toy_data_file = os.path.join(project_folder, "/tmp/tutorial5_data_bob1.pth")
crypten.save(toy_data, toy_data_file)
temp_files.append(toy_data_file)

import crypten.mpc as mpc
import crypten.communicator as comm


@mpc.run_multiprocess(world_size=2)
def forward_single_encrypted_layer():
    # Load and encrypt the layer
    layer = crypten.load_from_party(layer_linear_file, src=ALICE)
    layer_enc = crypten.nn.from_pytorch(layer, dummy_input=torch.empty((1, 4)))
    layer_enc.encrypt(src=ALICE)

    # Note that layer parameters are encrypted:
    crypten.print("Weights:\n", layer_enc.weight)
    crypten.print("Bias:\n", layer_enc.bias, "\n")

    # Load and encrypt data
    data_enc = crypten.load_from_party(toy_data_file, src=BOB)

    # Apply the encrypted layer (linear transformation):
    result_enc = layer_enc.forward(data_enc)

    # Decrypt the result:
    result = result_enc.get_plain_text()

    # Examine the result
    crypten.print("Decrypted result:\n", result)


forward_single_encrypted_layer()


# Initialize a linear layer with random weights
layer_scale = nn.Linear(3, 3)

# Construct a uniform scaling matrix: we'll scale by factor 5
factor = 5
layer_scale._parameters['weight'] = torch.eye(3)*factor
layer_scale._parameters['bias'] = torch.zeros_like(layer_scale._parameters['bias'])

# Save the plaintext layer
layer_scale_file = "/tmp/tutorial5_layer_alice2.pth"
crypten.save(layer_scale, layer_scale_file)
temp_files.append(layer_scale_file)

# Construct some toy data
features = 3
examples = 2
toy_data = torch.ones(examples, features)

# Save the plaintext toy data
toy_data_file = "/tmp/tutorial5_data_bob2.pth"
crypten.save(toy_data, toy_data_file)
temp_files.append(toy_data_file)


@mpc.run_multiprocess(world_size=2)
def forward_scaling_layer():
    rank = comm.get().get_rank()

    # Load and encrypt the layer
    layer = crypten.load_from_party(layer_scale_file, src=ALICE)
    layer_enc = crypten.nn.from_pytorch(layer, dummy_input=torch.empty((1, 3)))
    layer_enc.encrypt(src=ALICE)

    # Load and encrypt data
    data_enc = crypten.load_from_party(toy_data_file, src=BOB)

    # Note that layer parameters are (still) encrypted:
    crypten.print("Weights:\n", layer_enc.weight)
    crypten.print("Bias:\n\n", layer_enc.bias)

    # Apply the encrypted scaling transformation
    result_enc = layer_enc.forward(data_enc)

    # Decrypt the result:
    result = result_enc.get_plain_text()
    crypten.print("Plaintext result:\n", (result))


z = forward_scaling_layer()

# Setup
features = 50
examples = 1000

# Set random seed for reproducibility
torch.manual_seed(1)

# Generate toy data and separating hyperplane
data = torch.randn(examples, features)
w_true = torch.randn(1, features)
b_true = torch.randn(1)
labels = w_true.matmul(data.t()).add(b_true).sign()

# Change labels to non-negative values
labels_nn = torch.where(labels==-1, torch.zeros(labels.size()), labels)
labels_nn = labels_nn.squeeze().long()

# Split data into Alice's and Bob's portions:
data_alice, labels_alice = data[:900], labels_nn[:900]
data_bob, labels_bob = data[900:], labels_nn[900:]

# Define Alice's network
import torch.nn as nn
import torch.nn.functional as F


class AliceNet(nn.Module):
    def __init__(self):
        super(AliceNet, self).__init__()
        self.fc1 = nn.Linear(50, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out


# Train and save Alice's network
model = AliceNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for i in range(500):
    # forward pass: compute prediction
    output = model(data_alice)

    # compute and print loss
    loss = criterion(output, labels_alice)
    if i % 100 == 99:
        print("Epoch", i, "Loss:", loss.item())

    # zero gradients for learnable parameters
    optimizer.zero_grad()

    # backward pass: compute gradient with respect to model parameters
    loss.backward()

    # update model parameters
    optimizer.step()

sample_trained_model_file = '/tmp/tutorial5_alice_model.pth'
torch.save(model, sample_trained_model_file)
temp_files.append(sample_trained_model_file)


# Load the trained network to Alice
model_plaintext = crypten.load_from_party(sample_trained_model_file, model_class=AliceNet, src=ALICE)

# Convert the trained network to CrypTen network
private_model = crypten.nn.from_pytorch(model_plaintext, dummy_input=torch.empty((1, 50)))
# Encrypt the network
private_model.encrypt(src=ALICE)

# Examine the structure of the encrypted CrypTen network
for name, curr_module in private_model._modules.items():
    print("Name:", name, "\tModule:", curr_module)

# Pre-processing: Select only the first three examples in Bob's data for readability
data = data_bob[:3]
sample_data_bob_file = '/tmp/tutorial5_data_bob3.pth'
torch.save(data, sample_data_bob_file)
temp_files.append(sample_data_bob_file)


