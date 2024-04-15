import crypten
import torch

crypten.init()
torch.set_num_threads(1)

import subprocess
import os

project_folder = os.path.dirname(os.path.abspath(__file__))
tmp_dir = os.path.join(project_folder, "tmp")

# Run the mnist_utils.py script with the specified arguments
subprocess.run(['python3', './mnist_utils.py', '--option', 'train_v_test', '--dest', tmp_dir])

# Define Alice's network
import torch.nn as nn
import torch.nn.functional as F


class AliceNet(nn.Module):
    def __init__(self):
        super(AliceNet, self).__init__()
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


crypten.common.serial.register_safe_class(AliceNet)


def compute_accuracy(output, labels):
    pred = output.argmax(1)
    correct = pred.eq(labels)
    correct_count = correct.sum(0, keepdim=True).float()
    accuracy = correct_count.mul_(100.0 / output.size(0))
    return accuracy


ALICE = 0
BOB = 1

# Load pre-trained model to Alice
dummy_model = AliceNet()

model_path_alice = os.path.join(project_folder, "models", "tutorial4_alice_model.pth")

# Load the model
plaintext_model = torch.load(model_path_alice)

print(plaintext_model)

# Encrypt the model from Alice:

# 1. Create a dummy input with the same shape as the model input
dummy_input = torch.empty((1, 784))

# 2. Construct a CrypTen network with the trained model and dummy_input
private_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)

# 3. Encrypt the CrypTen network with src=ALICE
private_model.encrypt(src=ALICE)

# Check that model is encrypted:
print("Model successfully encrypted:", private_model.encrypted)

import crypten.mpc as mpc
import crypten.communicator as comm

labels = torch.load(tmp_dir + '/bob_test_labels.pth').long()
count = 100  # For illustration purposes, we'll use only 100 samples for classification


@mpc.run_multiprocess(world_size=2)
def encrypt_model_and_data():
    # Load pre-trained model to Alice
    model = crypten.load_from_party(model_path_alice, src=ALICE)

    # Encrypt model from Alice
    dummy_input = torch.empty((1, 784))
    private_model = crypten.nn.from_pytorch(model, dummy_input)
    private_model.encrypt(src=ALICE)
    #
    # model_path_bob = os.path.join(tmp_dir,'bob_test.pth')
    # # Load data to Bob
    # data_enc = crypten.load_from_party(model_path_bob, src=BOB)
    # data_enc2 = data_enc[:count]
    # data_flatten = data_enc2.flatten(start_dim=1)

    # Classify the encrypted data
    # private_model.eval()
    # output_enc = private_model(data_flatten)
    #
    # # Compute the accuracy
    # output = output_enc.get_plain_text()
    # accuracy = compute_accuracy(output, labels[:count])
    # crypten.print("\tAccuracy: {0:.4f}".format(accuracy.item()))


encrypt_model_and_data()


@mpc.run_multiprocess(world_size=2)
def encrypt_model_and_data():
    # Load pre-trained model to Alice
    plaintext_model = crypten.load_from_party(project_folder+'/models/tutorial4_alice_model.pth', src=ALICE)

    # Encrypt model from Alice
    dummy_input = torch.empty((1, 784))
    private_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)
    private_model.encrypt(src=ALICE)

    # Load data to Bob
    data_enc = crypten.load_from_party(project_folder+'/tmp/bob_test.pth', src=BOB)
    data_enc2 = data_enc[:count]
    data_flatten = data_enc2.flatten(start_dim=1)

    # Classify the encrypted data
    private_model.eval()
    output_enc = private_model(data_flatten)

    # Verify the results are encrypted:
    crypten.print("Output tensor encrypted:", crypten.is_encrypted_tensor(output_enc))

    # Decrypting the result
    output = output_enc.get_plain_text()

    # Obtaining the labels
    pred = output.argmax(dim=1)
    crypten.print("Decrypted labels:\n", pred)


encrypt_model_and_data()
