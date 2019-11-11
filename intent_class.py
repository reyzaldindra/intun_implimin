import torch
import torch.nn as nn
class LSTM_Intent(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units, output_size,n_layers):
        super(LSTM_Intent, self).__init__()
        self.hidden_units = hidden_units
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(p=0.4)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_units,self.n_layers)
        self.fc = nn.Linear(self.hidden_units, self.output_size)
        self.sig = nn.LogSoftmax(dim=1)
    def initialize_hidden_state(self):
        return torch.zeros((self.n_layers, 1, self.hidden_units)),torch.zeros((self.n_layers,1, self.hidden_units))
    
    def forward(self, x):
        ax = self.embedding(x)
        self.hidden = self.initialize_hidden_state()
        output, self.hidden = self.lstm(ax,self.hidden) # max_len X batch_size X hidden_units
        out = output
#         out = self.dropout(out)
        out = self.fc(out)
        out = self.dropout(out)
        out = out[-1, :,:]
        out = self.sig(out)
        return out