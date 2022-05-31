import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

torch.manual_seed(1)

#Here you can set the path where you want the graphs to be saved.
path = "C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/Graphs/"



################################################################################################
#This are the parameters of the model


T = 1 #Terminal time
sequence_len = 30 #Number of discretizations
dt = T/sequence_len
t1=np.arange(0,T,dt)
sqrdt = np.sqrt(dt)


initial_state = 1
drift = 0.1
volatility = 0.3
gamma = 0.2

jump_switch = True #Here you tell if you want the SDE to have jumps

rates = [10.0] #rate of the jump process
dim = len(rates) #dimension of the SDE, FOR NOW CODE ONLY WORKS FOR DIMENSION 1 !!!!


batch_size = 1000
hidden_dim = 512 #number of hidden parameters in one feedforward network


################################################################################################
#Here we compute the optimal control mathematically depending on whether we have jump or no

if jump_switch:
    coef = [gamma*volatility**2, rates[0]*gamma**2-drift*gamma+volatility**2, -drift]
    roots = np.roots(coef).tolist()
    opt_control = min(roots, key=abs)

else:
    opt_control = drift/volatility**2

##################################################################################
#Here we define our network. For the network scheme look at the picture (I will ad a cketch of the network cheme to the Git repository)

class ControlLSTM(nn.ModuleList):
    def __init__(self, sequence_len, dimension, hidden_dim, batch_size): #Network initialization
        super(ControlLSTM, self).__init__()

        # initialize the meta parameters
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.dimension = dimension

        # first layer lstm cell
        self.lstm_1 = nn.LSTMCell(input_size=dimension, hidden_size=hidden_dim)
        # fully connected layer to transform the dimension of the output of the LSTM cell to the dimension of the SDE (Output/control needs to be of the same dimention as SDE)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=dimension)
        self.activation = nn.Sigmoid()


    def forward(self, x, w,tj, hc): #function forwards defines the structure of our whole network (how we connect all the individual networks)

        # empty tensor for the output of the lstm, this is the contol
        output_seq = torch.empty((self.sequence_len,
                                  self.batch_size,
                                  self.dimension))

        # empty tensor for the inputs of the lstm, this is the wealth process
        input_seq = torch.empty((self.sequence_len,
                                  self.batch_size,
                                  self.dimension))



        # init. the both layer cells with the zeroth hidden and zeroth cell states (between LSTM Cells we pass information which is a tuple of hidden and cell state)

        hc_1 = hc #NOTE: This is a tuple


        # for every timestep use input x[t] to compute control out from hiden state h1 and derive the next imput x[t+1]
        for t in range(self.sequence_len):

            # get the hidden and cell states from the first layer cell, using current hidden-cell state and wealth
            hc_1 = self.lstm_1(x, hc_1)

            # unpack the hidden and the cell states from the first layer
            h_1, c_1 = hc_1

            #compute the output/control by leting the hidden state through fully connected layer. We don't use sigmoid activation since we allow control to be outside [0,1]
            out = self.fc(h_1)
            #out = self.activation(out)

            output_seq[t] = out #add output at time t to the output sequence
            if t < self.sequence_len - 1: #using the output/control compute the wealth at next timestep
                x = x + x * out * drift * dt + x * out * volatility * sqrdt * w[t] + x*out*gamma* tj[t]
            input_seq[t] = x #add wealth to wealth sequence

        # return the output sequence, terminal wealth and wealth sequence
        return output_seq, x, input_seq

    #functions that initialize hiden state, input, Brownian motion and Jump component of the process
    def init_hidden(self):
        # initialize the hidden state and the cell state to zeros
        return (torch.ones(self.batch_size, self.hidden_dim),
                torch.ones(self.batch_size, self.hidden_dim))

    def init_brownian(self):
        return torch.randn(self.sequence_len, self.batch_size, self.dimension)

    def init_state(self):
        #return torch.rand(self.batch_size, self.dimension)

        return torch.ones(self.batch_size, self.dimension)*initial_state

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

##################################################################################
#Custom loss function motivated by the log return at terminal time
# loss of the form -E[ln(|X_T|^2)]

def loss1(input):
    return - torch.mean(torch.log(torch.norm(input, dim=1)))


##################################################################################
#Learning phase

#Initialize the net
net = ControlLSTM(sequence_len=sequence_len, dimension=dim, hidden_dim=hidden_dim, batch_size=batch_size)

#Define the optimizer to be used
optimizer = optim.Adam(net.parameters(), lr=0.0005)



#Here we save the data obtained during the learning process
losses = []
controls = []
states = [] #terminal welth
state_seqs = [] #the whole wealth sequence

#Number of epochs/learning steps
epochs_number = 50

start = time.time() #Used to monitor the time needed

#Training loop
for epoch in range(epochs_number):
    print(f"Epoch {epoch}")

    #initialization
    hc = net.init_hidden()
    x = net.init_state()
    w = net.init_brownian()
    tj = net.init_jumpTimes()

    net.zero_grad() #set gradient to zero (pytorch remembers previous gradients

    control, state, state_seq = net(x, w, tj, hc) #feed the data through the net

    loss = loss1(state) #copute loss

    loss.backward() #backpropagation
    optimizer.step()

    #Save the data (.detach().cpu().numpy() is used to transform torch objects into numpy so we can plot them later

    #For controls, states and state_state_seqs we save the average over batch
    losses.append(loss.detach().cpu().numpy())
    controls.append(torch.mean(control[:,:,0], 1).detach().cpu().numpy())
    states.append(torch.mean(state[:,0]).detach().cpu().numpy())
    state_seqs.append(torch.mean(state_seq[:,:,0], 1).detach().cpu().numpy())

end= time.time()





##################################################################################

#Here we plot loss over the epochs (Remember -loss is exactly the expected log return at the terminal time)
epochs = np.arange(0,epochs_number,1)
plt.plot(epochs, losses)
plt.title("drift = " + str(drift) + ", vol = "+ str(volatility) + ", gamma = " + str(gamma) + ", epochs = "+ str(epochs_number) + ", batchsize = " + str(batch_size) + ", time = "+ str(int(end-start))+"s",fontsize= 10)
#plt.savefig(path + "loss"+ "d" +str(drift)+"v"+str(volatility)+"g"+str(gamma)+"e"+str(epochs_number)+"b"+str(batch_size)+".jpg")
plt.show()


#Here we plot how control changes over epochs. The red line represents mathematically determined optimal control
plt.plot(t1,controls[0],"palegreen")
plt.plot(t1,controls[int(epochs_number/5)],"azure")
plt.plot(t1,controls[int(epochs_number*2/5)],"lightblue")
plt.plot(t1,controls[int(epochs_number*3/5)], "silver")
plt.plot(t1,controls[int(epochs_number*4/5)], "dimgray")
plt.plot(t1,controls[-1],"black")
plt.axhline(y=opt_control, color='r', linestyle='-')
plt.title("drift = " + str(drift) + ", vol = "+ str(volatility) + ", gamma = " + str(gamma) + ", ep = "+ str(epochs_number) + ", bs = " + str(batch_size) + ", t = "+ str(int(end-start))+"s" + ", hd = " + str(hidden_dim),fontsize= 10)
#plt.savefig(path + "control"+ "d" +str(drift)+"v"+str(volatility) + "g"+ str(gamma)+"e" + str(200)+"b"+str(batch_size) + ", hd = " + str(hidden_dim)+".jpg")
plt.show()


#Here we plot a control process at the last epoch, sampled uniformly from the batch
i =np.random.randint(1000)
plt.plot(t1,control[:,i,0].detach().numpy(),"black")
plt.title("sdsad")
plt.show()

#Here we plot a wealth process at the last epoch, sampled uniformly from the batch
i =np.random.randint(1000)
plt.plot(t1, state_seq[:,1,0].detach().cpu().numpy())
plt.show()




