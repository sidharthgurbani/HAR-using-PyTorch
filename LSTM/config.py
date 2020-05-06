from model import LSTMModel, Bidir_LSTMModel

# This is the config file which holds the configuration values for different architectures

# Choose what architecure you want here:
arch = Architecture['LSTM1']

# This will choose the model also accordingly:
model = arch['model']

# This will set the values according to that architecture
bidir = arch['bidir']
clip_val = arch['clip_val']
drop_prob = arch['drop_prob']
n_epochs_hold = arch['n_epochs_hold']
n_layers = arch['n_layers']
learning_rate = arch['learning_rate']
weight_decay = arch['weight_decay']
n_highway_layers = arch['n_highway_layers']

# These are for diagnostics
diag = arch['diag']
save_file = arch['save_file']

# This will stay common for all architectures:
n_classes = 6
n_input = 9
n_hidden = 32
batch_size = 64
n_epochs = 100


# Configrations for different architectures explained below:

Architecture = {
	'LSTM1' : LSTM1,
	'LSTM2' : LSTM2,
	'Bidir_LSTM1' : Bidir_LSTM1,
	'Bidir_LSTM2' : Bidir_LSTM2
}

# Baseline LSTM Architecture:
LSTM1 = {
	'bidir' : False,
	'clip_val' : 10,
	'drop_prob' : 0.5,
	'n_epochs_hold' : 100,
	'n_layers' : 2,
	'learning_rate' : [0.0015],
	'weight_decay' : 0.001,
	'n_highway_layers' : 1,
	'model' : LSTMModel(),
	'diag' : 'Architecure chosen is baseline LSTM with 1 layer',
	'save_file' : 'results_lstm1.txt'
}

# Baseline LSTM with 2 layers Architecture:
LSTM2 = {
	'bidir' : False,
	'clip_val' : 10,
	'drop_prob' : 0.5,
	'n_epochs_hold' : 100,
	'n_layers' : 2,
	'learning_rate' : [0.0015],
	'weight_decay' : 0.001,
	'n_highway_layers' : 2,
	'model' : LSTMModel(),
	'diag' : 'Architecure chosen is baseline LSTM with 2 layers',
	'save_file' : 'results_lstm2.txt'
}
# Bidirectional LSTM Architecture:
Bidir_LSTM1 = {
	'bidir' : True,
	'clip_val' : 10,
	'drop_prob' : 0.5,
	'n_epochs_hold' : 100,
	'n_layers' : 2,
	'learning_rate' : [0.0015],
	'weight_decay' : 0.001,
	'n_highway_layers' : 1,
	'model' : Bidir_LSTMModel(),
	'diag' : 'Architecure chosen is bidirectional LSTM with 1 layer',
	'save_file' : 'results_bidir_lstm1.txt'
}

# Bidirectional LSTM with 2 layers Architecture:
Bidir_LSTM1 = {
	'bidir' : True,
	'clip_val' : 10,
	'drop_prob' : 0.5,
	'n_epochs_hold' : 100,
	'n_layers' : 2,
	'learning_rate' : [0.0015],
	'weight_decay' : 0.001,
	'n_highway_layers' : 2,
	'model' : Bidir_LSTMModel(),
	'diag' : 'Architecure chosen is bidirectional LSTM with 2 layers',
	'save_file' : 'results_bidir_lstm2.txt'
}