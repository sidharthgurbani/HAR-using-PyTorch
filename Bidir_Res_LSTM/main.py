import torch
import numpy as np
import matplotlib.pyplot as plt
from loadDataset import load_X, load_y
from train import train
import json
from model import BiDirResidual_LSTMModel, init_weights
import config as cfg

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
DATASET_PATH = "../../data/"

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

# LSTM Neural Network's internal structure

#n_hidden = int(config.get('InputParameters', 'n_hidden'))  # Hidden layer num of features
#n_classes = int(config.get('InputParameters', 'n_classes'))  # Total classes (should go up, or should go down)

n_hidden = cfg.n_hidden
n_classes = cfg.n_classes

epochs = cfg.n_epochs
learning_rate = cfg.learning_rate
weight_decay = cfg.weight_decay
clip_val = cfg.clip_val
# Training
# check if GPU is available

#train_on_gpu = torch.cuda.is_available()
if (torch.cuda.is_available() ):
    print('Training on GPU')
else:
    print('GPU not available! Training on CPU. Try to keep n_epochs very small')


def plot(x_arg, y_arg, y_arg_train, y_arg_test, label, lr):
    if label=='accuracy' or label=='loss':
        plt.figure()
        plt.plot(x_arg, y_arg_train, color='blue', label='train')
        plt.plot(x_arg, y_arg_test, color='red', label='test')
        plt.legend()
        plt.xlabel('epochs', fontsize=14)
        plt.ylabel( label + '(%)', fontsize=14)
        plt.title('Training and Test ' + label , fontsize=20)
        plt.savefig(label + '_' + str(epochs) + '_' + lr)
        plt.show()
    else:
        plt.figure()
        plt.plot(x_arg+1, y_arg)
        plt.legend()
        plt.xlabel('learning_rate', fontsize=14)
        plt.ylabel('Training loss', fontsize=14)
        plt.title('Train loss v/s learning_rate')
        plt.savefig(label + '_' + str(epochs) + '_' + lr + '.png')
        plt.show()


def saveResults(params: dict = {}):
    if params:
        with open("results/params.json", 'w+') as fp:
            json.dump(params.cpu().numpy(), fp)

def checkPlots():
    label = 'accuracy'
    epochs = '100'
    lr = '0.001'
    x_arg = np.arange(1, 101)
    y_arg = np.arange(100, 0, -1)
    plt.figure()
    plt.plot(x_arg, y_arg)
    plt.legend()
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel( 'Accuracy (%)', fontsize=14)
    plt.title('Training and Test accuracy' , fontsize=20)
    plt.savefig(label + str(epochs) + lr + '.png')
    plt.show()


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

    for lr in learning_rate:
        net = BiDirResidual_LSTMModel()
        net.apply(init_weights)
        params = train(net.float(), X_train, y_train, X_test, y_test, epochs=epochs, lr=lr, weight_decay=weight_decay, clip_val=clip_val)
        #saveResults(params)
        #train_losses, train_acc, test_losses, test_acc = train(net.float(), X_train, y_train, X_test, y_test, epochs=epochs)
        plot(params['epochs'], None, params['train_loss'], params['test_loss'], 'loss', str(lr))
        plot(params['epochs'], None, params['train_accuracy'], params['test_accuracy'], 'accuracy', str(lr))
        plot(params['lr'], params['train_loss'], None, None, 'loss_lr', str(lr))


main()
#checkPlots()
