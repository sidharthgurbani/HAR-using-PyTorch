import warnings
warnings.filterwarnings('ignore')

import configparser
import torch
from torch import nn
import torch.nn.functional as F


config = configparser.ConfigParser()
config.read('project.properties')
n_classes = int(config.get('InputParameters', 'n_classes'))
n_input = int(config.get('InputParameters', 'n_input'))
n_hidden = int(config.get('InputParameters', 'n_hidden'))

class LSTMModel(nn.Module):

    def __init__(self, n_input=n_input, n_hidden=n_hidden, n_layers=2,
                 n_classes=n_classes, drop_prob=0.5):
        super(LSTMModel, self).__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.n_input = n_input

        self.lstm = nn.LSTM(n_input, n_hidden, n_layers, dropout=self.drop_prob)
        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, hidden):
        x = x.permute(1, 0, 2)
        x, hidden = self.lstm(x, hidden)
        # print("length of x is {}".format(x.size()))
        x = self.dropout(x)
        out = x[-1]
        out = out.contiguous().view(-1, self.n_hidden)
        # print("length of out is {}".format(out.size()))
        out = self.fc(out)
        out = F.softmax(out)

        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initialize hidden state'''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        # if (train_on_gpu):
        if (torch.cuda.is_available() ):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden


def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)