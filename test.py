import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

T = 1
sequence_len = 100
dt = T/sequence_len
t1=np.arange(0,T,dt)
sqrdt = np.sqrt(dt)
drift = 0.08
volatility = 0.3

class ControlLSTM(nn.ModuleList):
    def __init__(self, sequence_len, dimension, hidden_dim, batch_size):
        super(ControlLSTM, self).__init__()

        # init the meta parameters
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.dimension = dimension

        # first layer lstm cell
        self.lstm_1 = nn.LSTMCell(input_size=dimension, hidden_size=hidden_dim)
        # fully connected layer to connect the output of the LSTM cell to the output
        self.fc = nn.Linear(in_features=hidden_dim, out_features=dimension)

    def forward(self, x, hc):
        # empty tensor for the output of the lstm, this is the contol
        output_seq = torch.empty((self.sequence_len,
                                  self.batch_size,
                                  self.dimension))
        # init the both layer cells with the zero hidden and zero cell states
        hc_1 = hc
        # for every timestep use input x[t] to compute control out from hiden state h1 and derive the next imput x[t+1]
        for t in range(self.sequence_len):
            # get the hidden and cell states from the first layer cell
            hc_1 = self.lstm_1(x[t], hc_1)
            # unpack the hidden and the cell states from the first layer
            h_1, c_1 = hc_1
            out = self.fc(h_1)
            output_seq[t] = out
        # return the output and state sequence
        return output_seq, x, out

    #functions that initialize hiden state, input and noise
    def init_hidden(self):
        # initialize the hidden state and the cell state to zeros
        return (torch.zeros(self.batch_size, self.hidden_dim),
                torch.zeros(self.batch_size, self.hidden_dim))

    def init_state(self):
        w = torch.randn(self.sequence_len, self.batch_size, self.dimension)
        x = torch.ones(self.sequence_len, self.batch_size, self.dimension)
        for t in range(self.sequence_len - 1):
            x[t+1,:,:] += x[t,:,:]*drift*dt + x[t,:,:]*volatility*sqrdt*w[t,:,:]
        return torch.ones(self.sequence_len, self.batch_size, self.dimension)

#Custom loss function motivated by the log return at terminal time
# loss of the form -E[ln(|X_T|^2)]
def loss1(input):
    tmp = torch.norm(input, dim=1)
    return torch.mean(tmp)

net = ControlLSTM(sequence_len=sequence_len, dimension=1, hidden_dim=512, batch_size=30)
optimizer = optim.Adam(net.parameters(), lr=0.001)

#Training loop
losses = []
controls = []

for epoch in range(10):
    hc = net.init_hidden()
    x = net.init_state()

    net.zero_grad()
    control, state, out = net(x, hc)

    last = state[-1,:,:]
    loss = loss1(out)
    loss.backward()
    optimizer.step()

    losses.append(loss)






a = torch.rand(3)
print(id(a))
b = a +1 # same python object
print(id(b))
b = b - 1
print(id(b)) # object reference changed!
a -= 1  # in-place operation doesn't change object
print(id(a))  # still the same object
a +=  a*b + a*3
print(id(a))
a
b
c = torch.mean(a)
print(id(c))













##################################################################################














