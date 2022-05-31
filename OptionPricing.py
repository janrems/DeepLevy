import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

torch.manual_seed(1)
path = "C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/Graphs/"

T = 1
sequence_len = 30
dt = T/sequence_len
t1=np.arange(0,T,dt)
sqrdt = np.sqrt(dt)
initial_state = 1
drift = 0.3
volatility = 0.3
gamma = 0.2
jump_switch = False
rates = [10.0]
dim = len(rates)
batch_size = 1000
hidden_dim = 512
F = 0.6
s0 = 0.5



##################################################################################

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

    def forward(self,input, w,tj, hc):
        # empty tensor for the output of the lstm, this is the contol
        output_seq = torch.empty((self.sequence_len,
                                  self.batch_size,
                                  self.dimension))

        input_seq = torch.empty((self.sequence_len,
                                  self.batch_size,
                                  self.dimension))

        stock_seq = torch.empty((self.sequence_len,
                                 self.batch_size,
                                 self.dimension))


        xtmp = self.fc(input)
        x = self.activation(xtmp)
        x0 = x

        #x0 = input
        #x = x0

        # init the both layer cells with the zeroth hidden and zeroth cell states
        hc_1 = hc
        s = torch.ones(self.batch_size, self.dimension) * s0
        stock_seq[0] = s
        input_seq[0] = x0
        # for every timestep use input x[t] to compute control out from hiden state h1 and derive the next imput x[t+1]
        for t in range(self.sequence_len):
            # get the hidden and cell states from the first layer cell
            hc_1 = self.lstm_1(x, hc_1)
            # unpack the hidden and the cell states from the first layer
            h_1, c_1 = hc_1
            out = self.fc(h_1)
            #out = self.activation(out)

            output_seq[t] = out
            if t < self.sequence_len - 1:
                x = x + x * out * drift * dt + x * out * volatility * sqrdt * w[t] + x*out*gamma* tj[t]
                s = s + s* drift * dt + s *  volatility * sqrdt * w[t] + s *gamma* tj[t]
                input_seq[t+1] = x
                stock_seq[t+1] = s
        # return the output and state sequence


        return output_seq, x, input_seq, x0, s, stock_seq

    #functions that initialize hiden state, input and noise
    def init_input(self):
        # initialize the hidden state and the cell state to zeros
        return torch.ones(self.batch_size, self.hidden_dim)

    def init_initial(self):
        return torch.rand(self.batch_size, self.dimension)



    def init_hidden(self):
        # initialize the hidden state and the cell state to zeros
        return (torch.rand(self.batch_size, self.hidden_dim),
                torch.rand(self.batch_size, self.hidden_dim))

    def init_brownian(self):
        return torch.randn(self.sequence_len, self.batch_size, self.dimension)


    def init_jumpTimes(self):
        tj = torch.zeros(self.sequence_len, self.batch_size, self.dimension)

        if jump_switch:

            for bn in range(self.batch_size):
                for dn in range(self.dimension):
                    cum_time = np.random.exponential(1 / rates[dn])
                    while (cum_time < T):
                        indx = int(cum_time / dt)
                        jumpsize = 1 - (2 * np.random.randint(2))  # enakomerno -1,1
                        tj[indx, bn, dn] += jumpsize

                        cum_time += np.random.exponential(1 / rates[dn])

        return tj


#computes the loss E[1/2 (X_T - F)^2]
def loss1(input):
    FF = torch.ones(batch_size, dim)*F
    return 0.5 * torch.mean(torch.square(torch.norm(input - FF, dim=1)))

def loss2(x,sT):
    Ftmp = torch.ones(batch_size, dim) * F
    FF = torch.max(torch.zeros(batch_size,dim), sT-Ftmp)
    return 0.5 * torch.mean(torch.square(torch.norm(x - FF, dim=1)))


##################################################################################

net = ControlLSTM(sequence_len=sequence_len, dimension=dim, hidden_dim=hidden_dim, batch_size=batch_size)
optimizer = optim.Adam(net.parameters(), lr=0.0005)

#Training loop
losses = []
controls = []
states = []
state_seqs = []
initials = []
stock_seqs = []

epochs_number = 300

start = time.time()
for epoch in range(epochs_number):
    print(f"Epoch {epoch}")
    hc = net.init_hidden()
    input = net.init_input()
    w = net.init_brownian()
    tj = net.init_jumpTimes()
    net.zero_grad()
    control, state, state_seq, initial, sT, stock_seq = net(input, w, tj, hc)


    #loss = loss1(state)
    loss = loss2(state,sT)
    loss.backward()
    optimizer.step()

    losses.append(loss.detach().cpu().numpy())
    controls.append(torch.mean(control[:,:,0], 1).detach().cpu().numpy())
    states.append(torch.mean(state[:,0]).detach().cpu().numpy())
    state_seqs.append(torch.mean(state_seq[:,:,0], 1).detach().cpu().numpy())
    initials.append(torch.mean(initial[:,0]).detach().cpu().numpy())
    stock_seqs.append(torch.mean(stock_seq[:,:,0], 1).detach().cpu().numpy())
end = time.time()





##################################################################################



time = -1
plt.plot(t1,state_seqs[time])
plt.show()




epochs = np.arange(0,epochs_number,1)
plt.plot(epochs,initials)
plt.show()


i =np.random.randint(1000)
opt = max(stock_seq[-1,i,0].detach().cpu().numpy()-F,0)
plt.plot(t1,control[:,i,0].detach().numpy(),"black")
plt.plot(t1, state_seq[:,i,0].detach().cpu().numpy(), "blue")
plt.plot(t1, stock_seq[:,i,0].detach().cpu().numpy())
plt.axhline(y=opt, color='r', linestyle='-')
plt.title("sdsad")
plt.show()


control[-1,i,0]
state_seq[0,i,0]



plt.plot(t1,controls[0],"palegreen")
plt.plot(t1,controls[int(epochs_number/5)],"azure")
plt.plot(t1,controls[int(epochs_number*2/5)],"lightblue")
plt.plot(t1,controls[int(epochs_number*3/5)], "silver")
plt.plot(t1,controls[int(epochs_number*4/5)], "dimgray")
plt.plot(t1,controls[-1],"black")
#plt.axhline(y=opt_control, color='r', linestyle='-')
plt.title("drift = " + str(drift) + ", vol = "+ str(volatility) + ", gamma = " + str(gamma) + ", ep = "+ str(epochs_number) + ", bs = " + str(batch_size) + ", t = "+ str(int(end-start))+"s" + ", hd = " + str(hidden_dim),fontsize= 10)
#plt.savefig(path + "control"+ "d" +str(drift)+"v"+str(volatility) + "g"+ str(gamma)+"e" + str(200)+"b"+str(batch_size) + ", hd = " + str(hidden_dim)+".jpg")

plt.show()


plt.plot(t1, state_seq[:,2,0].detach().cpu().numpy())
plt.show()

plt.plot(t1,state_seqs[4])
plt.show()

i = np.random.randint(1000)
plt.plot(t1, stock_seq[:,i,0].detach().cpu().numpy())
plt.show()

plt.plot(t1,controls[-5],"azure")
plt.plot(t1,controls[-4],"lightblue")
plt.plot(t1,controls[-3], "silver"); plt.plot(t1,controls[-2], "dimgray")
plt.plot(t1,controls[-1],"black")
plt.show()



plt.plot(epochs, losses)
#plt.title("drift = " + str(drift) + ", vol = "+ str(volatility) + ", gamma = " + str(gamma) + ", epochs = "+ str(epochs_number) + ", batchsize = " + str(batch_size) + ", time = "+ str(int(end-start))+"s",fontsize= 10)
#plt.savefig(path + "loss"+ "d" +str(drift)+"v"+str(volatility)+"g"+str(gamma)+"e"+str(epochs_number)+"b"+str(batch_size)+".jpg")
plt.show()

plt.plot(epochs, states)
plt.show()

plt.plot(epochs, -np.log(states))
plt.show()


plt.plot(epochs, losses)
plt.show()

plt.plot(t1,state_seq[:,0,0].detach().numpy())
plt.show()

a2 = volatility**2*gamma
a1 = gamm**2 - drift*gamma + volatility**2
a0 = - drift


##################################################################################

# net = ControlLSTM(sequence_len=sequence_len, dimension=dim, hidden_dim=512, batch_size=1000)
# optimizer = optim.Adam(net.parameters(), lr=0.0005)
#
# #Training loop
# losses = []
# controls = []
# states = []
# state_seqs = []
#
# e_numbers = 0
# condition = 1
# previous = torch.ones(sequence_len)*10
# while(condition>0.005):
#     #print(f"Epoch {epoch}")
#     hc = net.init_hidden()
#     x = net.init_state()
#     w = net.init_brownian()
#     tj = net.init_jumpTimes()
#     net.zero_grad()
#     control, state, state_seq = net(x, w, tj, hc)
#
#     loss = loss1(state)
#     loss.backward()
#     optimizer.step()
#
#     losses.append(loss.detach().cpu().numpy())
#     controls.append(torch.mean(control[:,:,0], 1).detach().cpu().numpy())
#     states.append(torch.mean(state[:,0]).detach().cpu().numpy())
#     state_seqs.append(torch.mean(state_seq[:,:,0], 1).detach().cpu().numpy())
#
#     new = torch.mean(control[:,:,0], 1)
#     condition = torch.norm(new-previous)
#     previous = new
#     e_numbers += 1
#     print(e_numbers)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
