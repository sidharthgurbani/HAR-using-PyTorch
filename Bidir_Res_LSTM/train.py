import torch
from torch import nn
import numpy as np
from test import test
from Functions import extract_batch_size, getLRScheduler
import torch.nn.utils.clip_grad as clip_grad
from sklearn.utils import shuffle
import config as cfg


batch_size = cfg.batch_size

def train(net, X_train, y_train, X_test, y_test, epochs=100, lr=0.001, weight_decay=0.005, clip_val=15):
    print("\n\n********** Running training! ************\n\n")
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = getLRScheduler(optimizer=opt)
    criterion = nn.CrossEntropyLoss()

    #if (train_on_gpu):
    if (torch.cuda.is_available() ):
        net.cuda()

    train_losses = []
    # results = np.empty([0, 5], dtype=np.float32)
    net.train()

    epoch_train_losses = []
    epoch_train_acc = []
    epoch_test_losses = []
    epoch_test_acc = []
    train_len = len(X_train)
    X_tr = X_train
    y_tr = y_train
    params = {
        'epochs' : [],
        'train_loss' : [],
        'test_loss' : [],
        'lr' : [],
        'train_accuracy' : [],
        'test_accuracy' : []
    }
    for epoch in range(epochs):
        train_losses = []
        step = 1

        h = net.init_hidden(batch_size)

        train_accuracy = 0
        #np.random.shuffle(X_tr)
        #np.random.shuffle(y_tr)

        while step * batch_size <= train_len:
            batch_xs = extract_batch_size(X_tr, step, batch_size)
            # batch_ys = one_hot_vector(extract_batch_size(y_train, step, batch_size))
            batch_ys = extract_batch_size(y_tr, step, batch_size)

            inputs, targets = torch.from_numpy(batch_xs), torch.from_numpy(batch_ys.flatten('F'))
            #if (train_on_gpu):
            if (torch.cuda.is_available() ):
                inputs, targets = inputs.cuda(), targets.cuda()

            h = tuple([each.data for each in h])
            opt.zero_grad()

            output = net(inputs.float(), h)
            # print("lenght of inputs is {} and target value is {}".format(inputs.size(), targets.size()))
            train_loss = criterion(output, targets.long())
            train_losses.append(train_loss.item())

            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == targets.view(*top_class.shape).long()
            train_accuracy += torch.mean(equals.type(torch.FloatTensor))
            equals = top_class

            train_loss.backward()
            clip_grad.clip_grad_norm_(net.parameters(), clip_val)
            opt.step()
            step += 1

        p = opt.param_groups[0]['lr']
        params['lr'].append(p)
        params['epochs'].append(epoch)
        sched.step()
        train_loss_avg = np.mean(train_losses)
        train_accuracy_avg = train_accuracy/(step-1)
        epoch_train_losses.append(train_loss_avg)
        epoch_train_acc.append(train_accuracy_avg)
        print("Epoch: {}/{}...".format(epoch + 1, epochs),
              ' ' * 16 + "Train Loss: {:.4f}".format(train_loss_avg),
              "Train accuracy: {:.4f}...".format(train_accuracy_avg))
        test_loss, test_f1score, test_accuracy = test(net, X_test, y_test, criterion, test_batch=batch_size)
        epoch_test_losses.append(test_loss)
        epoch_test_acc.append(test_accuracy)
        if ((epoch+1) % 10 == 0):
            print("Epoch: {}/{}...".format(epoch + 1, epochs),
                  ' ' * 16 + "Test Loss: {:.4f}...".format(test_loss),
                  "Test accuracy: {:.4f}...".format(test_accuracy),
                  "Test F1: {:.4f}...".format(test_f1score))

    params['train_loss'] = epoch_train_losses
    params['test_loss'] = epoch_test_losses
    params['train_accuracy'] = epoch_train_acc
    params['test_accuracy'] = epoch_test_acc
    return params
