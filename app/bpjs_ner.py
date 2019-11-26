import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)


    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        #print(sentence.shape)
        #self.hidden = self.init_hidden()
        embeds = self.word_embeddings(sentence)
        #print(embeds.shape)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, self.hidden_dim))
        #print(lstm_out.shape)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        #print(tag_space.shape)
        tag_scores = F.log_softmax(tag_space)
        return tag_scores
