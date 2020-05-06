import torch
import numpy as np
import sklearn.metrics as metrics
from Functions import extract_batch_size
import config as cfg
import copy

def test(net, X_test, y_test, criterion, best_accuracy, best_model, test_batch=64):
    
    net.eval()
    test_losses = []
    test_len = len(X_test)
    test_h = net.init_hidden(test_batch)
    test_accuracy = 0
    test_f1score = 0
    step = 1

    while step*test_batch <= test_len:
        batch_xs = extract_batch_size(X_test, step, test_batch)
        batch_ys = extract_batch_size(y_test, step, test_batch)

        inputs, targets = torch.from_numpy(batch_xs), torch.from_numpy(batch_ys.flatten('F'))
        #if (train_on_gpu):
        if (torch.cuda.is_available() ):
            inputs, targets = inputs.cuda(), targets.cuda()

        test_h = tuple([each.data for each in test_h])
        #print("Size of inputs is: {}".format(X_test.shape))
        output = net(inputs.float(), test_h)
        test_loss = criterion(output, targets.long())
        test_losses.append(test_loss.item())

        top_p, top_class = output.topk(1, dim=1)
        equals = top_class == targets.view(*top_class.shape).long()
        #print("\nDebugging here")
        #print(top_class.shape)
        #print(output.shape)
        #print(test_batch)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor))
        #print(test_accuracy)
        test_f1score += metrics.f1_score(top_class.cpu(), targets.view(*top_class.shape).long().cpu(), average='macro')
        step += 1

    test_loss_avg = np.mean(test_losses)
    test_f1_avg = test_f1score/(step-1)
    test_accuracy_avg = test_accuracy/(step-1)
    if (test_accuracy_avg > best_accuracy):
        best_accuracy = test_accuracy_avg
        best_model = copy.deepcopy(net)

    net.train()

    return test_loss_avg, test_f1_avg, test_accuracy_avg, best_accuracy, best_model
