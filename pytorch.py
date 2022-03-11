import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim



torch.manual_seed(1)

T = 1
sequence_len = 20
dt = T/sequence_len
t1=np.arange(0,T,dt)
sqrdt = np.sqrt(dt)
drift = 0.3
volatility = 0.2


class CharLSTM(nn.ModuleList):
    def __init__(self, sequence_len, vocab_size, hidden_dim, batch_size):
        super(CharLSTM, self).__init__()

        # init the meta parameters
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size

        # first layer lstm cell
        self.lstm_1 = nn.LSTMCell(input_size=vocab_size, hidden_size=hidden_dim)

        # second layer lstm cell
        #self.lstm_2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)

        # dropout layer for the output of the second layer cell
        #self.dropout = nn.Dropout(p=0.5)

        # fully connected layer to connect the output of the LSTM cell to the output
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(self, x, hc):
        """
            x: input to the model
                *  x[t] - input of shape (batch, input_size) at time t

            hc: hidden and cell states
                *  tuple of hidden and cell state
        """

        # empty tensor for the output of the lstm
        output_seq = torch.empty((self.sequence_len,
                                  self.batch_size,
                                  self.vocab_size))

        # pass the hidden and the cell state from one lstm cell to the next one
        # we also feed the output of the first layer lstm cell at time step t to the second layer cell
        # init the both layer cells with the zero hidden and zero cell states
        hc_1 = hc
        #hc_1, hc_2= hc, hc
        x =
        w = torch.randn(self.sequence_len, self.batch_size, self.vocab_size)

        # for every step in the sequence
        for t in range(self.sequence_len):
            # get the hidden and cell states from the first layer cell
            hc_1 = self.lstm_1(x[t], hc_1)

            # unpack the hidden and the cell states from the first layer
            h_1, c_1 = hc_1

            # pass the hidden state from the first layer to the cell in the second layer
            #hc_2 = self.lstm_2(h_1, hc_2)

            # unpack the hidden and cell states from the second layer cell
            #h_2, c_2 = hc_2

            # form the output of the fc
            out = self.fc(h_1)
            output_seq[t] = out
            if t < self.sequence_len - 1:
                x[t+1] = x[t] + x[t] * out * drift * dt + x[t] * out * volatility *sqrdt * w[t]

        # return the output and state sequence
        return output_seq, x

    def init_hidden(self):
        # initialize the hidden state and the cell state to zeros
        return (torch.zeros(self.batch_size, self.hidden_dim),
                torch.zeros(self.batch_size, self.hidden_dim))

    def init_brownian(self):
        return torch.randn(self.sequence_len, self.batch_size, self.vocab_size)

    def init_state(self):
        return torch.ones(self.sequence_len, self.batch_size, self.vocab_size)



# loss of the form -E[ln(|X|^2)]
def loss(input):
    l2 = torch.norm(input, dim=1)
    log = torch.log(l2)
    loss = - torch.mean(log)
    return loss


# compile the network - sequence_len, vocab_size, hidden_dim, batch_size
net = CharLSTM(sequence_len=sequence_len, vocab_size=1, hidden_dim=512, batch_size=30)

# define the loss and the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)


hc = net.init_hidden()



































