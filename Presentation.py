import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm

torch.manual_seed(1)
path = "C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/Graphs/"


###############################################################################

#Learning parameters NEEDS TO BE ALWAYS RUN!!!!!!
batch_size = 512
hidden_dim = 512
fixed_initial = False
merton_switch = False
jump_switch = False

#MODEL PARAMETERS NEEDS TO BE ALWAYS RUN
T = 1
sequence_len = 30
dt = T/sequence_len
t1=np.arange(0,T,dt)
sqrdt = np.sqrt(dt)



drift = 0.2
drift_orig = drift
volatility = 0.2
r = 0

F = 0.5
s0 = 1.0

#########################################################################################

#JUMP PARAMETERS
jump_switch = True
gamma = 1
rates = [20.0]
dim = len(rates)
drift = drift_orig - rates[0]*gamma

#MERTON PARAMETERS
merton_switch = True
mu = -0.2
sigma = 0.05
jump_rate = np.exp(mu + 0.5*sigma**2 ) - 1
drift = drift_orig - rates[0]*jump_rate #TODO: generalisation for multiple dimensions



#CASE OF FIXED INITIALS
fixed_initial = True




####################################################################################
# ANALYTIC RESULT

def BS(initial, strike, vol, terminal,rate):
    d1 = (1/(vol*np.sqrt(terminal)))*(np.log(initial/strike) + terminal*(rate+0.5*vol**2))
    d2 = d1- vol*np.sqrt(terminal)
    call = norm.cdf(d1)*initial - norm.cdf(d2)*strike*np.exp(-rate*terminal)
    return call




if jump_switch:
    rate2 = rates[0]*(1+jump_rate)
    option_value = 0
    for i in range(80):
        vol_i = volatility**2 + (i*sigma**2)/T
        r_i = - rates[0]*jump_rate + (i*np.log(1+jump_rate))/T
        option_value += BS(s0,F,vol_i,T,r_i)*(np.exp(-rate2*T) * (rate2*T)**i)/(np.math.factorial(i))
else:
    option_value = BS(s0,F,volatility,T,0)


##################################################################################
#SDES

#Geometric Levy Process
def glp(x,s,out,t):
    dx = x *( (1-out)* r + out * drift )* dt + x * out * volatility * sqrdt * w[t] + x*out*gamma* tj[t]
    ds = s* drift* dt + s *  volatility * sqrdt * w[t] + s *gamma* tj[t]
    return dx, ds







##################################################################################
#Initial value

def initial_val(n):
    bm = np.sqrt(T)*torch.randn(n)
    poisson = torch.poisson(rates[0]*torch.ones(n))

    s = s0*torch.exp((drift - 0.5*volatility**2)*T*torch.ones(n) + volatility*bm + np.log(1+gamma*jump_switch)*poisson)

    ftmp = torch.ones(n) * F
    f = torch.max(torch.zeros(n), s - ftmp)
    # Z*
    G = -drift_orig / (volatility ** 2 + jump_switch * rates[0] * gamma ** 2)
    if G*gamma*jump_switch <=-1:
        return print("G*gamaa < -1")
    ZT = np.exp(
        (-0.5 * (volatility ** 2) * G ** 2 - rates[0] * G * jump_switch * gamma) * T * torch.ones(n) +
        G * volatility * bm + jump_switch * np.log(1 + jump_switch*G * gamma) * poisson)

    # z hat
    return torch.mean(f * ZT)




option_value = initial_val(100000000)






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


        if fixed_initial:
            x = input*option_value
            x0 = x
        else:
            xtmp = self.fc(input)
           #x = self.activation(xtmp)
            x = xtmp
            x0 = x




        # init the both layer cells with the zeroth hidden and zeroth cell states
        hc_1 = hc
        s = torch.ones(self.batch_size, self.dimension) * s0
        stock_seq[0] = s
        input_seq[0] = x
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

                dx, ds = glp(x,s,out,t)

                x = x + dx
                s = s + ds
                input_seq[t+1] = x
                stock_seq[t+1] = s
        # return the output and state sequence



        return output_seq, x, input_seq, x0, s, stock_seq

    #functions that initialize hiden state, input and noise
    def init_input(self):
        # initialize the hidden state and the cell state to zeros
        return torch.ones(self.batch_size, self.hidden_dim)

    def init_initial(self):
        return torch.ones(self.batch_size, self.dimension)



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

                        #Different types of the jumps in the compound Poisson process
                        #jumpsize = 1 - (2 * np.random.randint(2))  # uniform {-1,1}
                        #jumpsize = np.random.normal(mu, sigma)   #Merton: lognormal
                        jumpsize = 1 #HPP

                        tj[indx, bn, dn] += jumpsize

                        cum_time += np.random.exponential(1 / rates[dn])

        return tj


#computes the loss E[1/2 (X_T - F)^2]
def loss1(input):
    FF = torch.ones(batch_size, dim)*F
    return 0.5 * torch.mean(torch.square(torch.norm(input - FF, dim=1)))

#European call option loss in min variance case
def loss2(x,sT):
    Ftmp = torch.ones(batch_size, dim) * F
    FF = torch.max(torch.zeros(batch_size,dim), sT-Ftmp)
    return 0.5 * torch.mean(torch.square(torch.norm(x - FF, dim=1)))



########################################################################

net = ControlLSTM(sequence_len=sequence_len, dimension=dim, hidden_dim=hidden_dim, batch_size=batch_size)
optimizer = optim.Adam(net.parameters(), lr=0.0005)

#Training loop
losses = []
controls = []
states = []
state_seqs = []
initials = []
stock_seqs = []
neg_val = []



epochs_number = 12000

start = time.time()
loss_min = 1
for epoch in range(epochs_number):
    print(f"Epoch {epoch}")
    hc = net.init_hidden()
    if fixed_initial:
        input = net.init_initial()
    else:
        input = net.init_input()

    w = net.init_brownian()
    tj = net.init_jumpTimes()
    net.zero_grad()
    control, state, state_seq, initial, sT, stock_seq = net(input, w, tj, hc)



    loss = loss2(state,sT)

    loss.backward()
    optimizer.step()

    losses.append(loss.detach().cpu().numpy())
    controls.append(torch.mean(control[:,:,0], 1).detach().cpu().numpy())
    states.append(torch.mean(state[:,0]).detach().cpu().numpy())
    state_seqs.append(torch.mean(state_seq[:,:,0], 1).detach().cpu().numpy())
    initials.append(torch.mean(initial[:,0]).detach().cpu().numpy())
    stock_seqs.append(torch.mean(stock_seq[:,:,0], 1).detach().cpu().numpy())


    if loss < loss_min:
        e_min = epoch
        l_min= loss
        con_min = control[:,:,0].detach().cpu().numpy()
        sta_min = state_seq[:,:,0].detach().cpu().numpy()
        sto_min = stock_seq[:,:,0].detach().cpu().numpy()
        in_min = initial[:,0].detach().cpu().numpy()
        loss_min = loss


    ########
    #check for negative stock values

    ss = stock_seq[:, :, 0].detach().cpu().numpy()
    sst = ss >= 0
    pozs = np.sum(np.prod(sst,0))
    neg_val.append(512-pozs)

    ########
end = time.time()


(end-start)/60/60


##################################################################################
#Saving the results

if jump_switch:
    drift = drift_orig

if fixed_initial:
    name = "initial_fixed_J"
else:
    name = "HPP_5_J"

if r != 0:
    name = "r"+str(r) + name



np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" + "losses",losses)
np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" + "initials",initials)
np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" + "stock", stock_seq[:,:,0].detach().cpu().numpy())
np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" + "control", control[:,:,0].detach().cpu().numpy())
np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" + "state", state_seq[:,:,0].detach().cpu().numpy())



torch.save(net.state_dict(), "C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"s" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_model_dic")
torch.save(net, "C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"s" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_model")

net = torch.load("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/JTruei0.5s0.2d3.804903871613452v0.2bs512ep5.0k_model")


###################################################################################
#GRAPHS

#Convergence of initials
epochs = np.arange(0,epochs_number,1)
epch = np.arange(0,4000,1)
plt.plot(epochs,initials)
plt.plot(epch, initials[8000:])
plt.axhline(y=option_value, color='r', linestyle='-')
plt.show()




#optimal portfolio
i =np.random.randint(batch_size)

x = sto_min[:,i]
K = np.ones(sequence_len)*F

d1 = (np.log(x/K) + (r + 0.5 * volatility**2)*(np.ones(sequence_len)*T - t1))/(volatility*np.sqrt(np.ones(sequence_len)*T - t1))

por = norm.cdf(d1)

ratio = por*x/sta_min[:,i]



#Market realisation
opt = max(sto_min[-1,i]-F,0)
plt.plot(t1,con_min[:,i],"black",label="Replicating portfolio")
plt.plot(t1,ratio,color="black", linestyle="--",label="BS replic. portfoli")
plt.plot(t1, sta_min[:,i], "blue", label="Wealth process")
plt.plot(t1, sto_min[:,i], label="Stock process")
plt.axhline(y=opt, color='r', linestyle='-', label="Option payoff")
plt.title("One market realisation at minimal loss epoch")
plt.legend(loc="center left")
plt.savefig(path + "OptionPricing/" +name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" +"market.jpg")
plt.show()





#Loss graph
plt.plot(epochs, losses)
plt.title("drift = " + str(drift) + ", vol = "+ str(volatility) + ", gamma = " + str(gamma) + ", epochs = "+ str(epochs_number) + ", batchsize = " + str(batch_size) + ", time = "+ str(int(end-start))+"s",fontsize= 10)
plt.savefig(path + "loss"+ "d" +str(drift)+"v"+str(volatility)+"g"+str(gamma)+"e"+str(epochs_number)+"b"+str(batch_size)+".jpg")
plt.plot(epochs,l_before[:epochs_number],color="r")
plt.show()

