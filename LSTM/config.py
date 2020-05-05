# This is the config file which holds the configuration values

n_classes = 6
n_input = 9
n_hidden = 32

learning_rate = 0.0015 #	[0.001, 0.0025, 0.005, 0.01]
weight_decay = 0.001
bidir = 2
clip_val = 20

batch_size = 64
drop_prob = 0.5
n_epochs = 100
n_layers = 2

n_epochs_hold = 100


# Baseline LSTM Architecture:
# bidir = 0
# clip_val = 10
# drop_prob = 0.5
# n_epochs_hol = 100
# n_layers = 2
# learning_rate = 0.0015 #	[0.001, 0.0025, 0.005, 0.01]
# weight_decay = 0.001

# Bidirectional LSTM Architecture:
# bidir = 1
# clip_val = 10
# drop_prob = 0.5
# n_epochs_hol = 100
# n_layers = 2
# learning_rate = 0.0015 #	[0.001, 0.0025, 0.005, 0.01]
# weight_decay = 0.001

# 2 layered Bidirectional LSTM Architecture:
