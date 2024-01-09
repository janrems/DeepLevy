import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
import torch.distributions as dist

torch.manual_seed(1)
path = "C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/Graphs/"


###############################################################################

#Learning parameters NEEDS TO BE ALWAYS RUN!!!!!!
batch_size = 1000
hidden_dim = 512
fixed_initial = False
merton_switch = False
jump_switch = False
mf_switch = False
bm2_switch = False
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
###############################################

#MERTON PARAMETERS
rates = [5.0]
jump_switch = True
merton_switch = True
gamma = 1
mu = -0.2
sigma = 0.05
jump_rate = np.exp(mu + 0.5*sigma**2 ) - 1 #k
drift = drift_orig - rates[0]*jump_rate #TODO: generalisation for multiple dimensions



def BS(initial, strike, vol, terminal,rate):
    d1 = (1/(vol*np.sqrt(terminal)))*(np.log(initial/strike) + terminal*(rate+0.5*vol**2))
    d2 = d1- vol*np.sqrt(terminal)
    call = norm.cdf(d1)*initial - norm.cdf(d2)*strike*np.exp(-rate*terminal)
    return call




if jump_switch:
    rate2 = rates[0]*(1+jump_rate)
    option_value = 0
    for i in range(80):
        #vol_i = volatility**2 + (i*sigma**2)/T
        vol_i = np.sqrt(volatility ** 2 + (i * sigma ** 2) / T)
        r_i = - rates[0]*jump_rate + (i*np.log(1+jump_rate))/T
        option_value += BS(s0,F,vol_i,T,r_i)*(np.exp(-rate2*T) * (rate2*T)**i)/(np.math.factorial(i))
else:
    option_value = BS(s0,F,volatility,T,0)



#Geometric Levy Process
def glp(x,s,out,t):
    dx = x *( (1-out)* r + out * drift )* dt + x * out * volatility * sqrdt * w[t] + x*out*gamma* tj[t]
    ds = s* drift* dt + s *  volatility * sqrdt * w[t] + s *gamma* tj[t]
    return dx, ds


w = torch.randn(sequence_len, batch_size,1)

tj = torch.zeros(sequence_len, batch_size,1)

tj2 = torch.zeros(sequence_len, batch_size,1)

for bn in range(batch_size):
    cum_time = np.random.exponential(1 / rates[0])
    while (cum_time < T):
        indx = int(cum_time / dt)
        jtmp = np.random.normal(mu, sigma)
        jumpsize = np.exp(jtmp) - 1  # Merton: lognormal

        tj[indx, bn] += jumpsize
        tj2[indx,bn] += jtmp
        cum_time += np.random.exponential(1 / rates[0])

output_seqM = torch.empty((sequence_len,batch_size,1))

input_seqM = torch.empty((sequence_len,batch_size,1))

stock_seqM = torch.empty((sequence_len,batch_size,1))

input_seqM2 = torch.empty((sequence_len,batch_size,1))

stock_seqM2 = torch.empty((sequence_len,batch_size,1))

x = torch.ones(batch_size,1)*option_value
x0 = x

y = torch.ones(batch_size,1)*option_value
y0 = y



K = torch.ones(batch_size,1)*F

s = torch.ones(batch_size,1) * s0

s2 = torch.ones(batch_size,1) * s0

stock_seqM[0] = s
input_seqM[0] = x

stock_seqM2[0] = s2
input_seqM2[0] = y

# for every timestep use input x[t] to compute control out from hiden state h1 and derive the next imput x[t+1]
for t in range(sequence_len):
    #Theoretical control
    #print("s= " + str(s[0, 0]))
    d1 = (torch.log(s / K) + (r + 0.5 * volatility ** 2) * (T-t*dt)*torch.ones(batch_size,1)) / (
               volatility * np.sqrt((T-t*dt)*torch.ones(batch_size,1)))

    #print("d1= " + str(d1[0,0]))
    n = dist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    por = n.cdf(d1)
    #print("por= " + str(por[0,0]))
    out = por*s/x
    #out = s/x

    output_seqM[t] = out
    if t < sequence_len - 1:
        dx, ds = glp(x,s,out,t)
        x = x + dx
        s = s + ds

        input_seqM[t+1] = x
        stock_seqM[t+1] = s

        ###########################################
        #using the solution
        w_cum = torch.cumsum(w*sqrdt,dim=0)
        tj_cum = torch.cumsum(tj2, dim=0)
        s2 = s0*np.exp((drift - 0.5*volatility**2)*(t+1)*dt*torch.ones(batch_size,1) + volatility*w_cum[t+1,:,:] + tj_cum[t+1,:,:])
        y  = y0*np.exp(out*(drift - 0.5*volatility**2)*(t+1)*dt*torch.ones(batch_size,1) + out*volatility*w_cum[t+1,:,:] + out*tj_cum[t+1,:,:])

        input_seqM2[t + 1] = y
        stock_seqM2[t + 1] = s2
# return the output and state sequence



i =np.random.randint(batch_size)
opt = max(stock_seqM[-1,i]-F,0)
plt.plot(t1,output_seqM[:,i,0],"black")
plt.plot(t1, input_seqM[:,i], "blue")
plt.plot(t1, stock_seqM[:,i,0])
#plt.plot(t1, input_seqM2[:,i], "green")
#plt.plot(t1, stock_seqM2[:,i,0],"brown")
#plt.plot(t1,w_cum[:,i,0])
#plt.plot(t1,tj_cum[:,i,0])
plt.axhline(y=opt, color='r', linestyle='-')
plt.title("sdsad")
plt.show()


it = input_seqM[-1,:,0].detach().cpu().numpy()
stt = stock_seqM[-1,:,0].detach().cpu().numpy()
k = np.ones(1000)*F
f = np.maximum.reduce([np.zeros(1000),stt-k])

err3 = it-f
plt.hist(err3, bins=np.linspace(-0.1,0.1,30))
plt.show()

err2 = (it-f)**2
np.average(err2)