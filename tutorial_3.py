import crypten
import torch

crypten.init()
torch.set_num_threads(1)

num_features = 100
num_train_examples = 1000
num_test_examples = 100
epochs = 40
lr = 3.0

# Set random seed for reproducibility
torch.manual_seed(1)

features = torch.randn(num_features, num_train_examples)
w_true = torch.randn(1, num_features)
b_true = torch.randn(1)

labels = w_true.matmul(features).add(b_true).sign()

test_features = torch.randn(num_features, num_test_examples)
test_labels = w_true.matmul(test_features).add(b_true).sign()

ALICE = 0
BOB = 1

from examples.mpc_linear_svm.mpc_linear_svm import train_linear_svm, evaluate_linear_svm

from crypten import mpc

import os

# Get the path of the current Python script's directory
project_folder = os.path.dirname(os.path.abspath(__file__))

# Add a new folder called "new_folder" to the path
tmp = os.path.join(project_folder, "tmp")

# Specify file locations to save each piece of data
filenames = {
    "features": os.path.join(tmp, "features.pth"),
    "labels": os.path.join(tmp, "labels.pth"),
    "features_alice": os.path.join(tmp, "features_alice.pth"),
    "features_bob": os.path.join(tmp, "features_bob.pth"),
    "samples_alice": os.path.join(tmp, "samples_alice.pth"),
    "samples_bob": os.path.join(tmp, "samples_bob.pth"),
    "w_true": os.path.join(tmp, "w_true.pth"),
    "b_true": os.path.join(tmp, "b_true.pth"),
    "test_features": os.path.join(tmp, "test_features.pth"),
    "test_labels": os.path.join(tmp, "test_labels.pth"),
}


@mpc.run_multiprocess(world_size=2)
def save_all_data():
    # Save features, labels for Data Labeling example
    crypten.save(features, filenames["features"])
    crypten.save(labels, filenames["labels"])

    # Save split features for Feature Aggregation example
    features_alice = features[:50]
    features_bob = features[50:]

    crypten.save_from_party(features_alice, filenames["features_alice"], src=ALICE)
    crypten.save_from_party(features_bob, filenames["features_bob"], src=BOB)

    # Save split dataset for Dataset Aggregation example
    samples_alice = features[:, :500]
    samples_bob = features[:, 500:]
    crypten.save_from_party(samples_alice, filenames["samples_alice"], src=ALICE)
    crypten.save_from_party(samples_bob, filenames["samples_bob"], src=BOB)

    # Save true model weights and biases for Model Hiding example
    crypten.save_from_party(w_true, filenames["w_true"], src=ALICE)
    crypten.save_from_party(b_true, filenames["b_true"], src=ALICE)

    crypten.save_from_party(test_features, filenames["test_features"], src=BOB)
    crypten.save_from_party(test_labels, filenames["test_labels"], src=BOB)


save_all_data()

from crypten import mpc


@mpc.run_multiprocess(world_size=2)
def data_labeling_example():
    """Apply data labeling access control model"""
    # Alice loads features, Bob loads labels
    features_enc = crypten.load_from_party(filenames["features"], src=ALICE)
    labels_enc = crypten.load_from_party(filenames["labels"], src=BOB)

    # Execute training
    w, b = train_linear_svm(features_enc, labels_enc, epochs=epochs, lr=lr)

    # Evaluate model
    evaluate_linear_svm(test_features, test_labels, w, b)


data_labeling_example()

@mpc.run_multiprocess(world_size=2)
def feature_aggregation_example():
    """Apply feature aggregation access control model"""
    # Alice loads some features, Bob loads other features
    features_alice_enc = crypten.load_from_party(filenames["features_alice"], src=ALICE)
    features_bob_enc = crypten.load_from_party(filenames["features_bob"], src=BOB)

    # Concatenate features
    features_enc = crypten.cat([features_alice_enc, features_bob_enc], dim=0)

    # Encrypt labels
    labels_enc = crypten.cryptensor(labels)

    # Execute training
    w, b = train_linear_svm(features_enc, labels_enc, epochs=epochs, lr=lr)

    # Evaluate model
    evaluate_linear_svm(test_features, test_labels, w, b)


feature_aggregation_example()


@mpc.run_multiprocess(world_size=2)
def dataset_augmentation_example():
    """Apply dataset augmentation access control model"""
    # Alice loads some samples, Bob loads other samples
    samples_alice_enc = crypten.load_from_party(filenames["samples_alice"], src=ALICE)
    samples_bob_enc = crypten.load_from_party(filenames["samples_bob"], src=BOB)

    # Concatenate features
    samples_enc = crypten.cat([samples_alice_enc, samples_bob_enc], dim=1)

    labels_enc = crypten.cryptensor(labels)

    # Execute training
    w, b = train_linear_svm(samples_enc, labels_enc, epochs=epochs, lr=lr)

    # Evaluate model
    evaluate_linear_svm(test_features, test_labels, w, b)


dataset_augmentation_example()


@mpc.run_multiprocess(world_size=2)
def model_hiding_example():
    """Apply model hiding access control model"""
    # Alice loads the model
    w_true_enc = crypten.load_from_party(filenames["w_true"], src=ALICE)
    b_true_enc = crypten.load_from_party(filenames["b_true"], src=ALICE)

    # Bob loads the features to be evaluated
    test_features_enc = crypten.load_from_party(filenames["test_features"], src=BOB)

    # Evaluate model
    evaluate_linear_svm(test_features_enc, test_labels, w_true_enc, b_true_enc)


model_hiding_example()

import os

for fn in filenames.values():
    if os.path.exists(fn): os.remove(fn)