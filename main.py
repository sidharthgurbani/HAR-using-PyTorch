import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from loadDataset import load_X, load_y
from train import train
from model import LSTMModel, init_weights

# Useful Constants

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

TRAIN = "train/"
TEST = "test/"
DATASET_PATH = ""

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

# LSTM Neural Network's internal structure

n_hidden = 32  # Hidden layer num of features
n_classes = 6  # Total classes (should go up, or should go down)

# Training

learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iters = training_data_count * 300  # Loop 300 times on the dataset
BATCH_SIZE = 1500
display_iter = 30000  # To show test set accuracy during training

# check if GPU is available

train_on_gpu = torch.cuda.is_available()
if (train_on_gpu):
    print('Training on GPU')
else:
    print('GPU not available! Training on CPU. Try to keep n_epochs very small')

def main():

    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    # Input Data

    training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
    test_data_count = len(X_test)  # 2947 testing series
    n_steps = len(X_train[0])  # 128 timesteps per series
    n_input = len(X_train[0][0])  # 9 input parameters per timestep


    # Some debugging info

    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("(X shape, y shape, every X's mean, every X's standard deviation)")
    print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
    print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

    net = LSTMModel()
    net.apply(init_weights)
    train(net.float())