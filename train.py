import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
import numpy as np
import test
from Functions import extract_batch_size

def train(net, X_train, y_train, X_test, y_test, epochs=100, lr=0.001):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if (train_on_gpu):
        net.cuda()

    train_losses = []
    results = np.empty([0, 5], dtype=np.float32)
    net.train()

    for epoch in range(epochs):
        epoch_loss = 0
        train_loss = 0
        train_sz = 0
        step = 1
        batch_size = BATCH_SIZE

        h = net.init_hidden(batch_size)

        train_len = len(X_train)

        while step * batch_size <= train_len:
            batch_xs = extract_batch_size(X_train, step, batch_size)
            # batch_ys = one_hot_vector(extract_batch_size(y_train, step, batch_size))
            batch_ys = extract_batch_size(y_train, step, batch_size)

            inputs, targets = torch.from_numpy(batch_xs), torch.from_numpy(batch_ys.flatten('F'))
            if (train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            h = tuple([each.data for each in h])

            opt.zero_grad()

            output, h = net(inputs.float(), h)
            # print("lenght of outputs is {} and target value is {}".format(output.size(), targets.size()))
            loss = criterion(output, targets.long())

            epoch_loss += loss.item()
            train_sz += batch_size

            loss.backward()
            opt.step()
            step += 1

        train_loss_avg = epoch_loss / train_sz
        print("Epoch: {}/{}...".format(epoch + 1, epochs),
              "Train Loss: {:.4f}".format(train_loss_avg))
        test_loss, test_f1score, test_accuracy = test(X_test, y_test, criterion)
        if (epoch % 10 == 0):
            print("Epoch: {}/{}...".format(epoch + 1, epochs),
                  ' ' * 16 + "Test Loss: {:.4f}...".format(test_loss),
                  "Test   accuracy: {:.4f}...".format(test_accuracy),
                  "Test F1: {:.4f}...".format(test_f1score))