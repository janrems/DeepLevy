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
batch_size = 256
hidden_dim = 512
fixed_initial = False
merton_switch = False
jump_switch = False
mf_switch = False
bm2_switch = False
#MODEL PARAMETERS NEEDS TO BE ALWAYS RUN
T = 1
sequence_len = 30
dt = T/sequence_len
t1=np.arange(0,T,dt)
sqrdt = np.sqrt(dt)



drift = 0.3
drift_orig = drift
volatility = 0.2
r = 0

gamma = 0
rates = [0]
dim = len(rates)

F = 0.5
s0 = 1.0


#########################################################################################
#Second brwonian motion added
bm2_switch = True
volatility2 = 0.3
batch_size = 512


#JUMP PARAMETER
jump_switch = True
gamma = 1
rates = [5.0]
dim = len(rates)
drift = drift_orig - rates[0]*gamma


#MERTON PARAMETERS
rates = [5.0]
jump_switch = True
merton_switch = True
gamma = 1
mu = -0.2
sigma = 0.05
jump_rate = np.exp(mu + 0.5*sigma**2 ) - 1 #k
drift = drift_orig - rates[0]*jump_rate #TODO: generalisation for multiple dimensions



#CASE OF FIXED INITIALS
fixed_initial = True


#MEAN FIELD CONTROL
mf_switch = True


####################################################################################
# ANALYTIC RESULT

def BS(initial, strike, vol, terminal, rate):
    d1 = (1/(vol*np.sqrt(terminal)))*(np.log(initial/strike) + terminal*(rate+0.5*vol**2))
    d2 = d1- vol*np.sqrt(terminal)
    call = norm.cdf(d1)*initial - norm.cdf(d2)*strike*np.exp(-rate*terminal)
    return call




if jump_switch:
    rate2 = rates[0]*(1+jump_rate)
    option_value = 0
    for i in range(80):
        vol_i = np.sqrt(volatility ** 2 + (i * sigma ** 2) / T)
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


# Mean-Field
def mf(x,out,t,bs,dim):
    ones = torch.ones(bs, dim)
    dx = torch.mean(x,dim=0) * torch.mean(out,dim = 0) * ones * dt +  volatility * sqrdt * w[t]
    ds = 0
    return dx, ds

def gbm2(x,s,out,t):
    dx = dx = x *out * drift * dt + x * out * volatility * sqrdt * w[t] + x*out*volatility2*sqrdt*tj[t]
    ds = s* drift* dt + s *  volatility * sqrdt * w[t] + s *volatility2*sqrdt*tj[t]
    return dx,ds






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


def initial_val_mert(n):
    k = np.exp(mu + 0.5 * sigma ** 2) - 1
    momen2 = np.exp(2 * mu + sigma ** 2) * (np.exp(sigma ** 2) - 1) + k ** 2
    G = -drift_orig / (volatility ** 2 + rates[0] * momen2)

    bm = np.sqrt(T)*torch.randn(n)
    poisson = np.random.poisson(rates[0], n)
    tjz = []
    tjs = []
    for i in poisson:
        logn = np.exp(np.random.normal(mu,sigma,i))
        transformedz = np.log(1+G*(logn-1))
        transformeds = np.log(logn)
        tjz.append(np.sum(transformedz))
        tjs.append(np.sum(transformeds))

    tjz = torch.tensor(tjz)
    tjs = torch.tensor(tjs)




    s = s0*torch.exp((drift - 0.5*volatility**2)*T*torch.ones(n) + volatility*bm + tjs)

    ftmp = torch.ones(n) * F
    f = torch.max(torch.zeros(n), s - ftmp)
    # Z*

    if torch.min(tjz) <=-1:
        return print("G*gamma < -1")

    ZT = torch.exp((-0.5 * (volatility ** 2) * G ** 2 - rates[0] * G * k) * T * torch.ones(n) + G * volatility * bm +  tjz)

    # z hat
    return torch.mean(f * ZT)

def initial_val_bm2(n):
    G = -drift_orig / (volatility ** 2 + volatility2**2)

    bm1 = np.sqrt(T) * torch.randn(n)
    bm2 = np.sqrt(T) * torch.randn(n)
    ZT = torch.exp((-0.5 * (volatility ** 2 + volatility2**2) * G ** 2) * T * torch.ones(n) + G * volatility * bm1 + G * volatility2 * bm2 )

    s = s0 * torch.exp((drift - 0.5 * (volatility ** 2 + volatility2**2)) * T * torch.ones(n) + volatility * bm1 + volatility2 * bm2 )

    ftmp = torch.ones(n) * F
    f = torch.max(torch.zeros(n), s - ftmp)

    return torch.mean(f * ZT)


def initial_val_mert_orig(n):
    k = np.exp(mu + 0.5 * sigma ** 2) - 1
    momen2 = np.exp(2 * mu + sigma ** 2) * (np.exp(sigma ** 2) - 1) + k ** 2
    G = -drift_orig / (volatility ** 2 + rates[0] * momen2)

    bm = np.sqrt(T)*torch.randn(n)
    poisson = np.random.poisson(rates[0], n)
    tjz = []
    tjs = []
    for i in poisson:
        logn = np.exp(np.random.normal(mu,sigma,i))
        transformedz = np.log(1+G*(logn-1))
        transformeds = np.log(logn)
        tjz.append(np.sum(transformedz))
        tjs.append(np.sum(transformeds))

    tjz = torch.tensor(tjz)
    tjs = torch.tensor(tjs)




    s = s0*torch.exp((drift - 0.5*volatility**2)*T*torch.ones(n) + volatility*bm + tjs)

    ftmp = torch.ones(n) * F
    f = torch.max(torch.zeros(n), s - ftmp)
    # Z*

    if torch.min(tjz) <=-1:
        return print("G*gamma < -1")

    ZT = torch.exp(-0.5 * ((drift_orig/volatility) ** 2) * T * torch.ones(n) - (drift_orig/volatility) * bm)

    # z hat
    return torch.mean(f * ZT)


op_v = initial_val_bm2(100000000)
print(op_v)



start = time.time()
option_value = initial_val_mert(500000)
end = time.time()



print(option_value)

ovorig = initial_val_mert_orig(1000000)
print(ovorig)






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
        self.lstm_1 = nn.LSTMCell(input_size=self.dimension, hidden_size=self.hidden_dim)
        #second layer lstm cell
        self.lstm_2 = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        # fully connected layer to connect the output of the LSTM cell to the output
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.dimension)
        self.activation = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.soft = nn.Softplus(beta=1,threshold=3)


    def forward(self,input, w,tj, hc1, hc2):
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
            x = self.soft(xtmp)
            #x = xtmp
            x0 = x




        # init the both layer cells with the zeroth hidden and zeroth cell states
        hc_1 = hc1
        hc_2 = hc2
        s = torch.ones(self.batch_size, self.dimension) * s0
        stock_seq[0] = s
        input_seq[0] = x
        # for every timestep use input x[t] to compute control out from hiden state h1 and derive the next imput x[t+1]
        for t in range(self.sequence_len):
            # get the hidden and cell states from the first layer cell
            hc_1 = self.lstm_1(x, hc_1)
            # unpack the hidden and the cell states from the first layer
            h_1, c_1 = hc_1
            hc_2 = self.lstm_2(h_1,hc_2)
            h_2, c_2 = hc_2
            out = self.fc(h_2)
            #out = self.activation(out)

            output_seq[t] = out
            if t < self.sequence_len - 1:
                if mf_switch == False:
                    if bm2_switch == False:
                        dx, ds = glp(x,s,out,t)
                    else:
                        dx,ds = gbm2(x,s,out,t)
                else:
                    dx, ds = mf(x,out,t,self.batch_size, self.dimension)
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
                        #jumpsize = np.random.normal(mu, sigma)   #Merton: normal
                        jumpsize = np.exp(np.random.normal(mu, sigma)) - 1 #Merton: lognormal
                        #jumpsize = 1 #HPP

                        tj[indx, bn, dn] += jumpsize

                        cum_time += np.random.exponential(1 / rates[dn])
        if bm2_switch:
            tj = torch.randn(self.sequence_len, self.batch_size, self.dimension)
        return tj


#computes the loss E[1/2 (X_T - F)^2]
def loss1(input):
    FF = torch.ones(batch_size, dim)*F
    return 0.5 * torch.mean(torch.square(torch.norm(input - FF, dim=1)))

def loss2(x,sT):
    Ftmp = torch.ones(batch_size, dim) * F
    FF = torch.max(torch.zeros(batch_size,dim), sT-Ftmp)
    return 0.5 * torch.mean(torch.square(torch.norm(x - FF, dim=1)))


def loss2put(x,sT):
    Ftmp = torch.ones(batch_size, dim) * F
    FF = torch.max(torch.zeros(batch_size,dim), Ftmp-sT)
    return 0.5 * torch.mean(torch.square(torch.norm(x - FF, dim=1)))


def loss3(x,x0,out_seq):
    Ftmp = torch.ones(batch_size, dim) * F
    terminal = torch.mean(torch.square(torch.norm(x - Ftmp, dim=1)))
    start = torch.mean(torch.square(torch.norm(x0, dim=1)))
    integral = torch.trapz(torch.square(torch.norm(out_seq,dim=2)),dx= dt,dim=0)
    ongoing = torch.mean(integral)
    return start + ongoing + terminal




########################################################################
name = "Layers2J"
name = "Layers2bm2"
name="BSretest"



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

st_seq = []
xt_seq = []

epochs_number = 3000
start = time.time()
loss_min = 1
for epoch in range(epochs_number):
    print(f"Epoch {epoch}")
    hc1 = net.init_hidden()
    hc2 = net.init_hidden()
    if fixed_initial:
        input = net.init_initial()
    else:
        input = net.init_input()

    w = net.init_brownian()
    tj = net.init_jumpTimes()
    net.zero_grad()
    control, state, state_seq, initial, sT, stock_seq = net(input, w, tj, hc1,hc2)

    ss = stock_seq[:, :, 0].detach().cpu().numpy()
    sst = ss >= 0
    pozs = np.sum(np.prod(sst, 0))
    if pozs < batch_size:
        #epochs_number += 1
        continue


    #loss = loss1(state)
    if mf_switch == False:
        loss = loss2(state,sT)
    else:
        loss = loss3(state,initial,control)
    loss.backward()
    optimizer.step()

    #losses.append(loss.detach().cpu().numpy())
    losses = np.append(losses,loss.detach().cpu().numpy())
    controls.append(torch.mean(control[:,:,0], 1).detach().cpu().numpy())
    states.append(torch.mean(state[:,0]).detach().cpu().numpy())
    state_seqs.append(torch.mean(state_seq[:,:,0], 1).detach().cpu().numpy())
    #initials.append(torch.mean(initial[:,0]).detach().cpu().numpy())
    initials = np.append(initials, torch.mean(initial[:,0]).detach().cpu().numpy())
    stock_seqs.append(torch.mean(stock_seq[:,:,0], 1).detach().cpu().numpy())


    st_seq.append(sT)
    xt_seq.append(state)

    if (loss < loss_min) and (epoch%50==0):
        e_min = epoch
        l_min= loss
        con_min = control[:,:,0].detach().cpu().numpy()
        sta_min = state_seq[:,:,0].detach().cpu().numpy()
        sto_min = stock_seq[:,:,0].detach().cpu().numpy()
        in_min = initial[:,0].detach().cpu().numpy()
        loss_min = loss
        net_min = net

    # if loss > 1.0 and epoch>300:
    #     oddstock = stock_seq[:,:,0].detach().cpu().numpy()
    #     oddwealth = state_seq[:,:,0].detach().cpu().numpy()
    #     oddcontrol = control[:,:,0].detach().cpu().numpy()
    #     strange = epoch
    #     break
    ########
    #check for negative stock values

    if ((epoch+1)%500==0):
        np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/" + name + str(jump_switch) + "i" + str(
            s0) + "F" + str(F) + "d" + str(drift_orig) + "v" + str(volatility) + "bs" + str(batch_size) + "ep" + str(
            epochs_number / 1000) + "k_" + "losses", losses)
        np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/" + name + str(jump_switch) + "i" + str(
            s0) + "F" + str(F) + "d" + str(drift_orig) + "v" + str(volatility) + "bs" + str(batch_size) + "ep" + str(
            epochs_number / 1000) + "k_" + "initials", initials)
        np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/" + name + str(jump_switch) + "i" + str(
            s0) + "F" + str(F) + "d" + str(drift_orig) + "v" + str(volatility) + "bs" + str(batch_size) + "ep" + str(
            epochs_number / 1000) + "k_" + "stock", sto_min)
        np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/" + name + str(jump_switch) + "i" + str(
            s0) + "F" + str(F) + "d" + str(drift_orig) + "v" + str(volatility) + "bs" + str(batch_size) + "ep" + str(
            epochs_number / 1000) + "k_" + "control", con_min)
        np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/" + name + str(jump_switch) + "i" + str(
            s0) + "F" + str(F) + "d" + str(drift_orig) + "v" + str(volatility) + "bs" + str(batch_size) + "ep" + str(
            epochs_number / 1000) + "k_" + "state", sta_min)

        torch.save(net_min.state_dict(),
                   "C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/" + name + str(jump_switch) + "i" + str(
                       s0) + "s" + str(F) + "d" + str(drift_orig) + "v" + str(volatility) + "bs" + str(
                       batch_size) + "ep" + str(epochs_number / 1000) + "k_model_dic")
        torch.save(net_min, "C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/" + name + str(jump_switch) + "i" + str(
            s0) + "s" + str(F) + "d" + str(drift_orig) + "v" + str(volatility) + "bs" + str(batch_size) + "ep" + str(
            epochs_number / 1000) + "k_model")


    ########
end = time.time()


(end-start)/60/60


##################################################################################
#Saving the results


if fixed_initial:
    name = "initial_fixed_J"
else:
    name = "HPP_5_J"

if r != 0:
    name = "r"+str(r) + name
if mf_switch:
    name = "MF-J"


name="logLay2Merton_J"

name = "Layers2J"

np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift_orig) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" + "losses",losses)
np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift_orig) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" + "initials",initials)
np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift_orig) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" + "stock", sto_min)
np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift_orig) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" + "control", con_min)
np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift_orig) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" + "state", sta_min)



torch.save(net.state_dict(), "C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"s" +str(F)+"d"+str(drift_orig) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_model_dic")
torch.save(net, "C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"s" +str(F)+"d"+str(drift_orig) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_model")

net = torch.load("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/Layers2JTruei1.0s0.5d0.2v0.2bs256ep5.0k_model")
###################################################################################

epochs = np.arange(0,epochs_number,1)

epochs = np.arange(0,len(initials),1)
plt.plot(epochs[:],initials[:])
plt.axhline(y=op_v, color='r', linestyle='-')
plt.show()


i =np.random.randint(batch_size)
opt = max(stock_seq[-1,i,0].detach().cpu().numpy()-F,0)
plt.plot(t1,control[:,i,0].detach().numpy(),"black")
plt.plot(t1, state_seq[:,i,0].detach().cpu().numpy(), "blue")
plt.plot(t1, stock_seq[:,i,0].detach().cpu().numpy())
plt.axhline(y=opt, color='r', linestyle='-')
plt.title("sdsad")
plt.show()



i =np.random.randint(batch_size)

x = sto_min[:,i]
K = np.ones(sequence_len)*F

d1 = (np.log(x/K) + (r + 0.5 * volatility**2)*(np.ones(sequence_len)*T - t1))/(volatility*np.sqrt(np.ones(sequence_len)*T - t1))

por = norm.cdf(d1)

ratio = por*x/sta_min[:,i]


i =np.random.randint(batch_size)
opt = max(sto_min[-1,i]-F,0)


plt.plot(t1,con_min[:,i],"black",label="Replicating portfolio")
#plt.plot(t1,ratio,color="black", linestyle="--",label="BS replic. portfoli")
plt.plot(t1, sta_min[:,i], "blue", label="Wealth process")
plt.plot(t1, sto_min[:,i], label="Stock process")
plt.axhline(y=opt, color='r', linestyle='-', label="Option payoff")
#plt.title("One market realisation at minimal loss epoch")
plt.legend(loc="center left")
plt.savefig(path + "OptionPricing/" +name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" +"market.jpg")
plt.show()




i =np.random.randint(batch_size)

x = stock[:,i]
K = np.ones(sequence_len)*F

d1 = (np.log(x/K) + (r + 0.5 * volatility**2)*(np.ones(sequence_len)*T - t1))/(volatility*np.sqrt(np.ones(sequence_len)*T - t1))

por = norm.cdf(d1)

ratio = por*x/state[:,i]




i =np.random.randint(batch_size)
opt = max(stock[-1,i]-F,0)
plt.plot(t1,control[:,i],"black",label="Replicating portfolio")
plt.plot(t1,ratio,color="black", linestyle="--",label="BS replicating portfolio")
plt.plot(t1, state[:,i], "blue", label="Wealth process")
plt.plot(t1, stock[:,i], label="Stock process")
plt.axhline(y=opt, color='r', linestyle='-', label="Option payoff")
#plt.title("One market realisation at minimal loss epoch")
plt.legend(loc="center left")
plt.savefig(path + "OptionPricing/Main/" +name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" +"market.jpg")
plt.show()


def distance():
    x = torch.tensor(stock)
    K = torch.ones((sequence_len, batch_size)) * F

    d1 = (torch.log(x / K) + (r + 0.5 * volatility ** 2) * (torch.ones((sequence_len,batch_size)) * T - torch.tensor(t1))) / (
                volatility * np.sqrt(np.ones((sequence_len, batch_size)) * T - t1))

    por = norm.cdf(d1)

    ratio = torch.tensor(por * x / state)
    port = torch.tensor(control)

    return torch.norm(ratio - port, dim = 0)

d = distance()


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



plt.plot(epochs[-1000:], losses[-1000:])
#plt.title("drift = " + str(drift) + ", vol = "+ str(volatility) + ", gamma = " + str(gamma) + ", epochs = "+ str(epochs_number) + ", batchsize = " + str(batch_size) + ", time = "+ str(int(end-start))+"s",fontsize= 10)
#plt.savefig(path + "loss"+ "d" +str(drift)+"v"+str(volatility)+"g"+str(gamma)+"e"+str(epochs_number)+"b"+str(batch_size)+".jpg")
plt.plot(epochs,l_before[:epochs_number],color="r")
plt.show()

plt.plot(eps2,losses[-10000:])
plt.show()


plt.plot(epochs, states)
plt.show()

plt.plot(epochs, -np.log(states))
plt.show()


plt.plot(epochs, losses,"r")
plt.plot(epochs,both_l)
plt.title("Red is loss for fixed initial wealt at 0.2")
plt.savefig(path+"loss_comparison.jpg")
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


zav = 0
for i in range(1000):

    x = np.array([-4.7, -0.7, -0.5, 3.1, 0.1])
    y = x-1
    t = np.dot(x-1,x-1)/4
    #print(t)
    if t < 1.145:
        zav += 1
print(zav/1000)

##################################################
toLoad = "Layers2JTruei1.0F0.5d0.2v0.2bs256ep5.0k_"


state = np.array(list(np.load("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/" + toLoad +"state.npy")))
control = np.array(list(np.load("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/" + toLoad +"control.npy")))
stock = np.array(list(np.load("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/" + toLoad +"stock.npy")))
initials = np.array(list(np.load("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/" + toLoad +"initials.npy")))
losses = np.array(list(np.load("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/" + toLoad +"losses.npy")))


i =np.random.randint(batch_size)

opt = max(stock[-1,i]-F,0)
plt.plot(t1,control[:,i],"black", label="Replicating portfolio")
plt.plot(t1, state[:,i], "blue",label="Wealth process")
plt.plot(t1, stock[:,i],label="Stock process")
plt.axhline(y=opt, color='r', linestyle='-', label = "Option payoff")
#plt.title("One market realisation at minimal loss epoch")
plt.legend(loc="upper left")
#plt.savefig(path + "OptionPricing/Main/" + toLoad + "market.jpg")
plt.show()

########################################################




#error distribution
xt = state[-1,:]
st = stock[-1,:]
k = np.ones(batch_size)*F
f = np.maximum.reduce([np.zeros(batch_size),st-k])

err = xt-f
plt.hist(err, bins=np.linspace(-0.1,0.1,30))
plt.show()

j = np.argmax(st-F)
np.max(-err)


err3 = (xt-f)**2
np.average(err3)


############################################
#direct from the loop
xts = torch.reshape(xt_seq[0],(-1,))
sts =torch.reshape(st_seq[0],(-1,))

for i in range(len(xt_seq)):
    if i != 0:

        xts = torch.cat((xts,torch.reshape(xt_seq[i],(-1,))))
        sts = torch.cat((sts,torch.reshape(st_seq[i],(-1,))))

xts = xts.detach().cpu().numpy()
sts = sts.detach().cpu().numpy()

l = len(xts)
k = np.ones(l)*F
f = np.maximum.reduce([np.zeros(l),sts-k])

err = xts-f
plt.hist(err, bins=np.linspace(-0.1,0.1,30), alpha=0.5, label="Our approach")
plt.hist(err3, bins=np.linspace(-0.1,0.1,30), alpha=0.5,label="Merton")
plt.legend(loc='upper right')
plt.savefig(path + "OptionPricing/Main/" +name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" +"lossdist.jpg")
plt.show()

np.max(err)
np.argmax(err)
########################################################






epochs_number = len(losses)
epochs = np.arange(0,len(initials),1)

first_n = 200


plt.plot(epochs[:first_n],initials[:first_n])
plt.axhline(y=option_value, color='r', linestyle='--', label="BS Option price")
plt.legend(loc="upper right")
plt.title("Initial wealth over " + str(first_n) + " epochs")
plt.savefig(path + "OptionPricing/" + toLoad +"first" +str(first_n) +"initials.jpg")
plt.show()


plt.plot(epochs[20:len(initials)],initials[20:len(initials)])
plt.axhline(y=option_value, color='r', linestyle='-')
plt.savefig(path + "OptionPricing/Main/" + toLoad + "initials.jpg")
plt.show()

np.mean(initials[-50:])

plt.plot(epochs[:first_n],losses[:first_n])
plt.title("Loss over " + str(first_n) + " epochs")
plt.savefig(path + "OptionPricing/" + toLoad + "first" +str(first_n) +"losses.jpg")
plt.show()

plt.plot(epochs[50:len(initials)],losses[50:len(initials)])
plt.savefig(path + "OptionPricing/Main/" + toLoad + "losses.jpg")
plt.show()

plt.plot(epochs[50:],losses[50:])
#plt.savefig(path + "OptionPricing/" + toLoad + "losses.jpg")
plt.show()

np.mean(losses[-50:])


abs(option_value - initials[-1])/s0


#########
x = stock[:,i]
K = np.ones(sequence_len)*F

d1 = (np.log(x/K) + (r + 0.5 * volatility**2)*(np.ones(sequence_len)*T - t1))/(volatility*np.sqrt(np.ones(sequence_len)*T - t1))

por = norm.cdf(d1)

ratio = por*x/state[:,i]

#L2  distance

w = torch.tensor(state)
st = torch.tensor(stock)
con = torch.tensor(control)

K = torch.ones(sequence_len, batch_size)*F

tis = torch.tensor(t1)
t_copy = torch.ones(sequence_len, batch_size)*tis.unsqueeze(1)

ter = torch.ones(sequence_len, batch_size)*T


d1 = (torch.log(st/K) + (ter-t_copy)*0.5*volatility**2)/(volatility*torch.sqrt(ter-t_copy))

n = torch.distributions.normal.Normal(0,1)

por = n.cdf(d1)

ratio = por*st/w

mse = torch.mean(((ratio-con).pow(2).sum(0)*dt).sqrt())


j = np.argmin(losses)
initials[j]
losses[j]


l2 = ((ratio-con).pow(2).sum(0)*dt).sqrt()

rat = ratio.detach().cpu().numpy()

i = np.random.randint(0,batch_size)

plt.plot(t1,rat[:,i])
plt.show()





###########
#Compare Z*, Z^M

time = torch.arange(0,1,sequence_len)

brow = torch.randn(sequence_len, batch_size)

poiss = torch.zeros(sequence_len, batch_size)

k = np.exp(mu + 0.5 * sigma ** 2) - 1
momen2 = np.exp(2 * mu + sigma ** 2) * (np.exp(sigma ** 2) - 1) + k ** 2
G = -drift_orig / (volatility ** 2 + rates[0] * momen2)

for bn in range(batch_size):
    cum_time = np.random.exponential(1 / rates[0])
    while (cum_time < T):
        indx = int(cum_time / dt)

        #Different types of the jumps in the compound Poisson process
        #jumpsize = 1 - (2 * np.random.randint(2))  # uniform {-1,1}
        #jumpsize = np.random.normal(mu, sigma)   #Merton: normal
        jumpsize = np.exp(np.random.normal(mu, sigma)) - 1 #Merton: lognormal
        #jumpsize = 1 #HPP

        poiss[indx, bn] += jumpsize

        cum_time += np.random.exponential(1 / rates[0])



Zs = torch.ones(sequence_len, batch_size)
Zm = torch.ones(sequence_len, batch_size)
# for every timestep use input x[t] to compute control out from hiden state h1 and derive the next imput x[t+1]
for t in range(sequence_len-1):
    Zm[t+1] = Zm[t] + Zm[t] *(-drift_orig/volatility * sqrdt * brow[t])
    Zs[t+1] = Zs[t] + Zs[t] * (-G*rates[0]*k * dt + G*volatility * sqrdt * brow[t] + G*poiss[t])



i = np.random.randint(batch_size)

plt.plot(t1, Zs[:,i].detach().cpu().numpy(), "blue", label= r'$Z^*$')
plt.plot(t1, Zm[:,i].detach().cpu().numpy(), "red", label=r'$Z^M$')
plt.legend(loc="upper right")
plt.savefig(path + "OptionPricing/Main/" +name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" +"Zs.jpg")
plt.show()


############################################################


def merOp(rate,mean,sd,vol):
    k = np.exp(mean + 0.5 * sd ** 2) - 1
    rate2 = rate * (1 + k)
    option_value = 0
    for i in range(80):
        # vol_i = volatility**2 + (i*sigma**2)/T
        vol_i = np.sqrt(vol ** 2 + (i * sd ** 2) / T)
        r_i = - rate * k + (i * np.log(1 + k)) / T
        option_value += BS(s0, F, vol_i, T, r_i) * (np.exp(-rate2 * T) * (rate2 * T) ** i) / (np.math.factorial(i))

    return option_value


def ourOp(n,rate,mean,sd,vol):
    k = np.exp(mean + 0.5 * sd ** 2) - 1
    momen2 = np.exp(2 * mean + sd ** 2) * (np.exp(sd ** 2) - 1) + k ** 2
    G = -drift_orig / (vol ** 2 + rate * momen2)


    bm = np.sqrt(T)*torch.randn(n)
    poisson = np.random.poisson(T*rate, n)
    tjz = []
    tjs = []
    for i in poisson:
        logn = np.exp(np.random.normal(mean,sd,i))
        transformedz = np.log(1+G*(logn-1))
        transformeds = np.log(logn)
        tjz.append(np.sum(transformedz))
        tjs.append(np.sum(transformeds))

    tjz = torch.tensor(tjz)
    tjs = torch.tensor(tjs)




    s = s0*torch.exp(((drift_orig - k* rate) - 0.5*vol**2)*T*torch.ones(n) + vol*bm + tjs)

    ftmp = torch.ones(n) * F
    f = torch.max(torch.zeros(n), s - ftmp)
    # Z*

    if torch.min(tjz) <=-1:
        return print("G*gamma < -1")

    ZT = torch.exp((-0.5 * (vol ** 2) * G ** 2 - rate * G * k) * T * torch.ones(n) + G * vol * bm +  tjz)

    # z hat
    op_val = torch.mean(f * ZT)
    return op_val.detach().cpu().numpy()



n_points = 21

lambdas = np.linspace(0,10,n_points)

mus = np.linspace(-0.5,-0.1, n_points)

sds = np.linspace(0,0.1, n_points)

vols = np.linspace(0,0.4, n_points)



mer_op = []

our_op = []


mer_op_m = []

our_op_m = []


mer_op_s = []

our_op_s = []


mer_op_v = []

our_op_v = []




for l in lambdas2:
    mer_op.append(merOp(l,mu, sigma,volatility))
    our_op.append(ourOp(500000,l,mu,sigma,volatility))



for m in mus:
    mer_op_m.append(merOp(rates[0],m, sigma,volatility))
    our_op_m.append(ourOp(1000000,rates[0],m,sigma,volatility))


for s in sds:
    mer_op_s.append(merOp(rates[0],mu, s,volatility))
    our_op_s.append(ourOp(500000,rates[0],mu,s,volatility))


for v in vols:
    mer_op_v.append(merOp(rates[0],mu, sigma, v))
    our_op_v.append(ourOp(500000,rates[0],mu,sigma,v))




plt.plot(lambdas2,mer_op,color="green")
plt.plot(lambdas2,our_op)
plt.show()


plt.plot(mus,mer_op_m,color="green")
plt.plot(mus,our_op_m)
plt.show()


plt.plot(sds,mer_op_s,color="green")
plt.plot(sds,our_op_s)
plt.show()


plt.plot(vols,mer_op_v,color="green")
plt.plot(vols,our_op_v)
plt.show()



start = time.time()
ourOp(1000000,rates[0],mu,sigma,volatility)
end = time.time()
cas = end-start



def lam_mu(n,s,v):
    lams, ms = np.meshgrid(lambdas, mus)
    p_m = np.zeros((n_points,n_points))
    p_o = np.zeros((n_points,n_points))
    i = 0
    for m in mus:
        print(m)
        j=0
        for l in lambdas:
            p_m[i,j] = merOp(l,m,s,v)
            p_o[i,j] = ourOp(n,l,m,s,v)
            j+=1
        i+=1

    return lams,ms,p_m,p_o

ls,ms,p_m_lm,p_o_lm = lam_mu(1000000,sigma,volatility)

np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+"prices"+"_drift"+str(drift_orig) +"v" + str(volatility) +"sigma" + str(sigma) +"lm_merton",p_m_lm)
np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+"prices"+"_drift"+str(drift_orig) +"v" + str(volatility) +"sigma" + str(sigma) +"lm_our",p_o_lm)

ls,ms = np.meshgrid(lambdas, mus)


def vol_sig(n,l,m):
    vs, ss = np.meshgrid(vols, sds)
    p_m = np.zeros((n_points,n_points))
    p_o = np.zeros((n_points,n_points))
    i = 0
    for v in vols:
        print(v)
        j=0
        for s in sds:
            p_m[i,j] = merOp(l,m,s,v)
            p_o[i,j] = ourOp(n,l,m,s,v)
            j+=1
        i+=1

    return vs,ss,p_m,p_o

vs,ss,p_m_vs,p_o_vs = vol_sig(1000000,rates[0],mu)

np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+"prices"+"_drift"+str(drift_orig) +"l" + str(rates[0]) +"mu" + str(mu) +"vs_merton",p_m_vs)
np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+"prices"+"_drift"+str(drift_orig) +"v" + str(volatility) +"sigma" + str(sigma) +"vs_our",p_o_vs)


p_m_lm = np.load("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/prices_drift0.2v0.2sigma0.05lm_merton.npy")
p_o_lm = np.load("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/prices_drift0.2v0.2sigma0.05lm_our.npy")


max_cases_lm = np.maximum(p_m_lm, p_o_lm)
max_cases_vs = np.maximum(p_m_vs, p_o_vs)
max_lm = np.max(max_cases_lm)
max_vs = np.max(max_cases_vs)

min_cases_lm = np.minimum(p_m_lm, p_o_lm)
min_cases_vs = np.minimum(p_m_vs, p_o_vs)
min_lm = np.min(min_cases_lm)
min_vs = np.min(min_cases_vs)
#####################################
p_m2 = p_m_lm[:-1, :-1]

fig, ax = plt.subplots()

c = ax.pcolormesh(ls, ms, p_m2, cmap='RdBu', vmin=min_lm, vmax=max_lm)

# set the limits of the plot to the limits of the data
ax.axis([ls.min(), ls.max(), ms.min(), ms.max()])
ax.set_xlabel(r'$\lambda$', fontsize=14)
ax.set_ylabel(r'$\mu$',fontsize=14)
plt.savefig(path + "OptionPricing/Main/Prices_M_lm.jpg")
fig.colorbar(c, ax=ax)

plt.show()
##############

p_o2 = p_o_lm[:-1, :-1]

fig, ax = plt.subplots()

c = ax.pcolormesh(ls, ms, p_o2, cmap='RdBu', vmin=min_lm, vmax=max_lm)

# set the limits of the plot to the limits of the data
ax.axis([ls.min(), ls.max(), ms.min(), ms.max()])
ax.set_xlabel(r'$\lambda$', fontsize=14)
ax.set_ylabel(r'$\mu$',fontsize=14)
fig.colorbar(c, ax=ax)
plt.savefig(path + "OptionPricing/Main/Prices_O_lm.jpg")
plt.show()

###########

diflm = p_o_lm[:-1, :-1] - p_m_lm[:-1,:-1]

diflm_min, diflm_max = np.min(diflm), np.max(diflm)

fig, ax = plt.subplots()

c = ax.pcolormesh(ls, ms, diflm, cmap='RdBu', vmin=diflm_min, vmax=diflm_max)

# set the limits of the plot to the limits of the data
ax.axis([ls.min(), ls.max(), ms.min(), ms.max()])
ax.set_xlabel(r'$\lambda$', fontsize=14)
ax.set_ylabel(r'$\mu$',fontsize=14)
fig.colorbar(c, ax=ax)
plt.savefig(path + "OptionPricing/Main/Prices_dif_lm.jpg")
plt.show()

#############################################################################

p_m3 = p_m_vs[:-1, :-1]

fig, ax = plt.subplots()

c = ax.pcolormesh(ss, vs, p_m3, cmap='RdBu', vmin=min_vs, vmax=max_vs)

# set the limits of the plot to the limits of the data
ax.axis([ss.min(), ss.max(), vs.min(), vs.max()])
ax.set_xlabel(r'$\delta$', fontsize=14)
ax.set_ylabel(r'$\sigma$',fontsize=14)
fig.colorbar(c, ax=ax)
plt.savefig(path + "OptionPricing/Main/Prices_M_vs.jpg")
plt.show()
##############

p_o3 = p_o_vs[:-1, :-1]

fig, ax = plt.subplots()

c = ax.pcolormesh(ss, vs, p_o3, cmap='RdBu', vmin=min_vs, vmax=max_vs)

# set the limits of the plot to the limits of the data
ax.axis([ss.min(), ss.max(), vs.min(), vs.max()])
ax.set_xlabel(r'$\delta$', fontsize=14)
ax.set_ylabel(r'$\sigma$',fontsize=14)
fig.colorbar(c, ax=ax)
plt.savefig(path + "OptionPricing/Main/Prices_O_vs.jpg")
plt.show()

###########

difvs = p_o_vs[:-1, :-1] - p_m_vs[:-1,:-1]

difvs_min, difvs_max = np.min(difvs), np.max(difvs)

fig, ax = plt.subplots()

c = ax.pcolormesh(ss, vs, difvs, cmap='RdBu', vmin=difvs_min, vmax=difvs_max)

# set the limits of the plot to the limits of the data
ax.axis([ss.min(), ss.max(), vs.min(), vs.max()])
ax.set_xlabel(r'$\delta$', fontsize=14)
ax.set_ylabel(r'$\sigma$',fontsize=14)
fig.colorbar(c, ax=ax)
plt.savefig(path + "OptionPricing/Main/Prices_dif_vs.jpg")
plt.show()














