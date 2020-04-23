import warnings
warnings.filterwarnings('ignore')

import torch
import sklearn.metrics as metrics

def test(X_test, y_test, criterion):

    test_h = net.init_hidden(len(X_test))
    net.eval()

    inputs, targets = torch.from_numpy(X_test), torch.from_numpy(y_test.flatten('F'))
    if (train_on_gpu):
        inputs, targets = inputs.cuda(), targets.cuda()

    test_h = tuple([each.data for each in test_h])
    print("Size of inputs is: {}".format(X_test.shape))
    output, test_h = net(inputs.float(), test_h)
    test_loss = criterion(output, targets.long())

    top_p, top_class = output.topk(1, dim=1)
    equals = top_class == targets.view(*top_class.shape).long()
    test_accuracy = torch.mean(equals.type(torch.FloatTensor))
    test_f1score = metrics.f1_score(top_class.cpu(), targets.view(*top_class.shape).long().cpu(), average='macro')

    net.train()

    return test_loss, test_f1score, test_accuracy