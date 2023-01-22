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
#MODEL PARAMETERS NEEDS TO BE ALWAYS RUN
T = 1
sequence_len = 150
dt = T/sequence_len
t1=np.arange(0,T,dt)
sqrdt = np.sqrt(dt)



drift = 0.2
drift_orig = drift
volatility = 0.2
r = 0

gamma = 0
rates = [0]
dim = len(rates)

F = 0.5
s0 = 1.0


#########################################################################################

#JUMP PARAMETER
jump_switch = True
gamma = 1
rates = [5.0]
dim = len(rates)
drift = drift_orig - rates[0]*gamma


#MERTON PARAMETERS
rates = [20.0]
jump_switch = True
merton_switch = True
gamma = 1
mu = -0.2
sigma = 0.05
jump_rate = np.exp(mu + 0.5*sigma**2 ) - 1
drift = drift_orig - rates[0]*jump_rate #TODO: generalisation for multiple dimensions



#CASE OF FIXED INITIALS
fixed_initial = True


#MEAN FIELD CONTROL
mf_switch = True


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
    ds = s* drift* dt + s *  volatility * sqrdt * w[t] + s *gamma* (torch.exp(tj[t])-torch.ones(batch_size,dim) )
    dl = (drift -0.5*volatility**2)*dt + volatility*sqrdt*w[t] + gamma*tj[t]
    return dx, ds, dl


# Mean-Field
def mf(x,out,t,bs,dim):
    ones = torch.ones(bs, dim)
    dx = torch.mean(x,dim=0) * torch.mean(out,dim = 0) * ones * dt +  volatility * sqrdt * w[t]
    ds = 0
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
        self.lstm_1 = nn.LSTMCell(input_size=self.dimension, hidden_size=self.hidden_dim)
        #second layer lstm cell
        self.lstm_2 = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        # fully connected layer to connect the output of the LSTM cell to the output
        self.lstm_3 = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.lstm_4 = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.dimension)
        self.activation = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.soft = nn.Softplus(beta=1,threshold=3)


    def forward(self,input, w,tj, hc1, hc2, hc3,hc4):
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

        s_seq = torch.empty((self.sequence_len,
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
        hc_3 = hc3
        hc_4 = hc4
        s = torch.ones(self.batch_size, self.dimension) * s0
        l= torch.zeros(self.batch_size, self.dimension)
        stock_seq[0] = s
        s_seq[0] = s
        input_seq[0] = x
        # for every timestep use input x[t] to compute control out from hiden state h1 and derive the next imput x[t+1]
        for t in range(self.sequence_len):
            # get the hidden and cell states from the first layer cell
            out = x*0
            #out = self.activation(out)

            output_seq[t] = out
            if t < self.sequence_len - 1:
                if mf_switch == False:
                    dx, ds, dl = glp(x,s,out,t)

                else:
                    dx, ds = mf(x,out,t,self.batch_size, self.dimension)
                x = x + dx
                s = s + ds
                l = l +dl

                input_seq[t+1] = x
                stock_seq[t+1] = s
                s_seq[t+1] = s0*torch.exp(l)
        # return the output and state sequence



        return output_seq, x, input_seq, x0, s, stock_seq, s_seq

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
                        jumpsize = np.random.normal(mu, sigma)   #Merton: lognormal
                        #jumpsize = np.exp(np.random.normal(mu, sigma)) - 1 #merton test
                        #jumpsize = 1 #HPP

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



epochs_number = 13000
start = time.time()
loss_min = 1
for epoch in range(epochs_number):
    print(f"Epoch {epoch}")
    hc1 = net.init_hidden()
    hc2 = net.init_hidden()
    hc3 = net.init_hidden()
    hc4 = net.init_hidden()
    if fixed_initial:
        input = net.init_initial()
    else:
        input = net.init_input()

    w = net.init_brownian()
    tj = net.init_jumpTimes()
    net.zero_grad()
    control, state, state_seq, initial, sT, stock_seq, s_seq = net(input, w, tj, hc1,hc2,hc3,hc4)


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


    if loss < loss_min:
        e_min = epoch
        l_min= loss
        con_min = control[:,:,0].detach().cpu().numpy()
        sta_min = state_seq[:,:,0].detach().cpu().numpy()
        sto_min = stock_seq[:,:,0].detach().cpu().numpy()
        in_min = initial[:,0].detach().cpu().numpy()
        loss_min = loss

    # if loss > 1.0 and epoch>300:
    #     oddstock = stock_seq[:,:,0].detach().cpu().numpy()
    #     oddwealth = state_seq[:,:,0].detach().cpu().numpy()
    #     oddcontrol = control[:,:,0].detach().cpu().numpy()
    #     strange = epoch
    #     break
    ########
    #check for negative stock values

    ss = stock_seq[:, :, 0].detach().cpu().numpy()
    sst = ss >= 0
    pozs = np.sum(np.prod(sst,0))
    neg_val.append(batch_size-pozs)



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
if mf_switch:
    name = "MF-J"


name="Lay4Merton_J"

name = "Layers2J"

np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" + "losses",losses)
np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" + "initials",initials)
np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" + "stock", stock_seq[:,:,0].detach().cpu().numpy())
np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" + "control", control[:,:,0].detach().cpu().numpy())
np.save("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" + "state", state_seq[:,:,0].detach().cpu().numpy())



torch.save(net.state_dict(), "C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"s" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_model_dic")
torch.save(net, "C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/"+name+str(jump_switch)+"i"+str(s0) +"s" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_model")

net = torch.load("C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/data/JFalsei1.0s0.5d0.3v0.2bs512ep12.0k_model")
###################################################################################

epochs = np.arange(0,epochs_number,1)

plt.plot(epochs,initials)

plt.axhline(y=option_value, color='r', linestyle='-')
plt.show()


i =np.random.randint(batch_size)
plt.plot(t1, stock_seq[:,i,0].detach().cpu().numpy())
plt.plot(t1, s_seq[:,i,0].detach().cpu().numpy())

plt.title("sdsad")
plt.show()


i =np.random.randint(batch_size)

x = sto_min[:,i]
K = np.ones(sequence_len)*F

d1 = (np.log(x/K) + (r + 0.5 * volatility**2)*(np.ones(sequence_len)*T - t1))/(volatility*np.sqrt(np.ones(sequence_len)*T - t1))

por = norm.cdf(d1)

ratio = por*x/sta_min[:,i]



opt = max(sto_min[-1,i]-F,0)
plt.plot(t1,con_min[:,i],"black",label="Replicating portfolio")
plt.plot(t1,ratio,color="black", linestyle="--",label="BS replic. portfoli")
plt.plot(t1,por,color="green", linestyle="--",label="BS rewplic. portfoli")
plt.plot(t1, sta_min[:,i], "blue", label="Wealth process")
plt.plot(t1, sto_min[:,i], label="Stock process")
plt.axhline(y=opt, color='r', linestyle='-', label="Option payoff")
plt.title("One market realisation at minimal loss epoch")
plt.legend(loc="center left")
#plt.savefig(path + "OptionPricing/" +name+str(jump_switch)+"i"+str(s0) +"F" +str(F)+"d"+str(drift) +"v" + str(volatility) +"bs" + str(batch_size) +"ep" + str(epochs_number/1000)+"k_" +"market.jpg")
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



plt.plot(epochs[200:], losses[200:])
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
toLoad = "logLay2Merton_JTruei1.0F0.5d0.2v0.2bs256ep14.0k_"


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
plt.title("One market realisation at minimal loss epoch")
plt.legend(loc="upper left")
plt.savefig(path + "OptionPricing/" + toLoad + "market.jpg")
plt.show()



epochs_number = len(in2)
epochs = np.arange(0,epochs_number,1)

first_n = 200

plt.plot(epochs[:],initials[:len(in2)])
plt.plot(epochs[:],in2, color="g")
plt.axhline(y=option_value, color='r', linestyle='-')
plt.savefig(path + "OptionPricing/" + toLoad + "initials.jpg")
plt.show()


plt.plot(epochs[:first_n],initials[:first_n])
plt.axhline(y=option_value, color='r', linestyle='--', label="BS Option price")
plt.legend(loc="upper right")
plt.title("Initial wealth over " + str(first_n) + " epochs")
plt.savefig(path + "OptionPricing/" + toLoad +"first" +str(first_n) +"initials.jpg")
plt.show()


plt.plot(epochs[:],initials[:])
plt.axhline(y=option_value, color='r', linestyle='-')
plt.savefig(path + "OptionPricing/" + toLoad + "initials.jpg")
plt.show()

np.mean(initials[-50:])

plt.plot(epochs[:first_n],losses[:first_n])
plt.title("Loss over " + str(first_n) + " epochs")
plt.savefig(path + "OptionPricing/" + toLoad + "first" +str(first_n) +"losses.jpg")
plt.show()

plt.plot(epochs[:],losses[:])
plt.savefig(path + "OptionPricing/" + toLoad + "losses.jpg")
plt.show()

np.mean(losses[-50:])


abs(option_value - initials[-1])/s0




T = 1
sequence_len = 1000000
dt = T/sequence_len
t=np.arange(0,T,dt)
sqrdt = np.sqrt(dt)

t[-1]
nor = np.random.normal(0,1,sequence_len)

bg = np.zeros(sequence_len)
for i in range(sequence_len - 1):
    bg[i + 1] = bg[i] + sqrdt * nor[i]

plt.plot(t, bg)
plt.show()


mu = 0.3
sigma = 0.2
s = np.zeros(sequence_len)
s[0] = 1
for i in range(sequence_len-1):
    s[i+1] = s[i] + mu*s[i]*dt + sigma*s[i]*(bg[i+1] - bg[i])

s2 = 1*np.exp((mu-0.5*sigma**2)*t + sigma*bg)

plt.plot(t,s)
plt.plot(t,s2,color="r")
plt.show()











x = np.zeros(sequence_len)
x[0] = 0
int = 0
x2 = np.zeros(sequence_len)
x2[0] = 0
for i in range(sequence_len-1):
    x[i+1] = x[i] + (sqrdt*nor[i])/(1-t[i])
    x2[i+1] = x2[i] -x2[i]/(1-t[i])*dt + sqrdt*nor[i]
x = (1-t)*x


y = bg - t*bg[-1]


plt.plot(t,x)
plt.plot(t,y, color="black")
plt.plot(t,x2, color= "g")
plt.axhline(y=0,color="r")
plt.show()



#####
t2 = t[0::100]
len(t2)
bg2 = bg[0::100]
dx = x2[0::100]

ys = y[0::100]

y2 = bg2-t2*bg2[-1]

dx2 = np.zeros(10000)
for i in range(10000-1):
    dx2[i + 1] = dx2[i] - dx2[i] / (1 - t[i]) * dt + (bg2[i+1]-bg2[i])


plt.plot(t2,dx, color="black")
plt.plot(t2,dx2, color= "g")
plt.axhline(y=0,color="r")
plt.show()

plt.plot(t2,y2, color="black")
plt.plot(t2,ys, color= "g")
plt.axhline(y=0,color="r")
plt.show()