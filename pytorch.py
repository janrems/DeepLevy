import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

T = 1
sequence_len = 30
dt = T/sequence_len
t1=np.linspace(0,T,sequence_len)
sqrdt = np.sqrt(dt)
drift = 0.05
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
        self.activation = nn.Sigmoid()

    def forward(self, x, w, hc):
        # empty tensor for the output of the lstm, this is the contol
        output_seq = torch.empty((self.sequence_len,
                                  self.batch_size,
                                  self.dimension))

        input_seq = torch.empty((self.sequence_len,
                                  self.batch_size,
                                  self.dimension))

        # init the both layer cells with the zero hidden and zero cell states
        hc_1 = hc
        # for every timestep use input x[t] to compute control out from hiden state h1 and derive the next imput x[t+1]
        for t in range(self.sequence_len):
            # get the hidden and cell states from the first layer cell
            hc_1 = self.lstm_1(x, hc_1)
            # unpack the hidden and the cell states from the first layer
            h_1, c_1 = hc_1
            out = self.fc(h_1)
            #out = self.activation(out)
            input_seq[t] = x
            output_seq[t] = out
            if t < self.sequence_len - 1:
                x = x + x * out * drift * dt + x * out * volatility * sqrdt * w[t]
        # return the output and state sequence
        return output_seq, x, input_seq

    #functions that initialize hiden state, input and noise
    def init_hidden(self):
        # initialize the hidden state and the cell state to zeros
        return (torch.ones(self.batch_size, self.hidden_dim),
                torch.ones(self.batch_size, self.hidden_dim))

    def init_brownian(self):
        return torch.randn(self.sequence_len, self.batch_size, self.dimension)

    def init_state(self):
        return torch.ones(self.batch_size, self.dimension)

#Custom loss function motivated by the log return at terminal time
# loss of the form -E[ln(|X_T|^2)]
def loss1(input):
    return - torch.mean(torch.log(torch.norm(input, dim=1)))

net = ControlLSTM(sequence_len=sequence_len, dimension=1, hidden_dim=512, batch_size=1000)
optimizer = optim.Adam(net.parameters(), lr=0.0005)

#Training loop
losses = []
controls = []
states = []
state_seqs = []

epochs_number = 30

for epoch in range(epochs_number):
    print(f"Epoch {epoch}")
    hc = net.init_hidden()
    x = net.init_state()
    w = net.init_brownian()
    net.zero_grad()
    control, state, state_seq = net(x, w, hc)

    loss = loss1(state)
    loss.backward()
    optimizer.step()

    losses.append(loss.detach().cpu().numpy())
    controls.append(torch.mean(control[:,:,0], 1).detach().cpu().numpy())
    states.append(torch.mean(state[:,0]).detach().cpu().numpy())
    state_seqs.append(torch.mean(state_seq[:,:,0], 1).detach().cpu().numpy())


##################################################################################

time = -1
plt.plot(t1,state_seqs[time])
plt.show()

plt.plot(t1,controls[25])
plt.plot(t1,controls[10])
plt.plot(t1,controls[20])
plt.plot(t1,controls[19])
plt.plot(t1,controls[12])
plt.plot(t1,controls[-1])
plt.show()


epochs = np.arange(0,epochs_number,1)
plt.plot(epochs, states)
plt.show()

plt.plot(epochs, -np.log(states))
plt.show()


plt.plot(epochs, losses)
plt.show()


plt.plot(t1,ss[:,2].detach().numpy())
plt.show()

plt.plot(t1, sm.detach().numpy())
plt.show()














