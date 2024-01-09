import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

from click.core import batch
from scipy.stats import norm
import torch.distributions as dist

torch.manual_seed(1)
path = "C:/Users/jan1r/Documents/Faks/Doktorat/DeepLevy/Graphs/"

n = dist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
###############################################################################

#Learning parameters NEEDS TO BE ALWAYS RUN!!!!!!
batch_size = 10
hidden_dim = 512
fixed_initial = False
merton_switch = False
jump_switch = False
mf_switch = False
bm2_switch = False
#MODEL PARAMETERS NEEDS TO BE ALWAYS RUN
T = 1
sequence_len = 30000
dt = T/sequence_len
t1=np.arange(0,T,dt)
sqrdt = np.sqrt(dt)



drift = 0.2
drift_orig = drift
volatility = 0.2
r = 0

gamma = 0
rates = [5.0]
dim = len(rates)
l = rates[0]

F = 0.5
s0 = 1.0


#########################################################################################

#MERTON PARAMETERS
rates = [5.0]
mu = -0.2
sigma = 0.05
jump_rate = np.exp(mu + 0.5*sigma**2 ) - 1 #k
drift = drift_orig - rates[0]*jump_rate #TODO: generalisation for multiple dimensions

k = np.exp(mu + 0.5 * sigma ** 2) - 1
momen2 = np.exp(2 * mu + sigma ** 2) * (np.exp(sigma ** 2) - 1) + k ** 2
G = -drift_orig / (volatility ** 2 + rates[0] * momen2)



###############################################################################################

ones = torch.ones(sequence_len, batch_size,1)

time = torch.from_numpy(t1)
time = time.unsqueeze(0).unsqueeze(-1)
time = time.repeat(batch_size, 1,1)

# repeat tensor along new dimension
time = torch.transpose(time, 0, 1)

###########################################################

def gen_beta(s,number=20):
    L = s*np.exp((drift_orig - 0.5*volatility**2 - l*k)*(T*ones-time))
    out = 0
    for j in range(number):
        ev = j*mu*ones
        var = ones*j*sigma**2+ (T*ones-time)*volatility**2
        f1 = torch.exp(ev+0.5*var)
        arg = ((ev + var)-torch.log(F*ones) + torch.log(L))/torch.sqrt(var)
        f2 = n.cdf(arg)
        f3 = (torch.exp(-l*(T*ones-time))*(l*(T*ones-time))**j)/(np.math.factorial(j))
        out += f1*f2*f3
    return volatility*L*out




def gen_kappa(s,y,number=20):
    L = s*np.exp((drift_orig - 0.5*volatility**2 - l*k)*(T*ones-time))
    out = 0
    for j in range(number):
        ev = j * mu * ones
        var = ones * j * sigma ** 2 + (T * ones - time) * volatility ** 2
        arg1 = (torch.log(F*ones) - torch.log(y*L) - ev)/torch.sqrt(var)
        s1 = n.cdf(arg1)
        arg2 = (torch.log(F*ones) - torch.log(L) - ev)/torch.sqrt(var)
        s2 = n.cdf(arg2)
        f3 = (torch.exp(-l*(T*ones-time))*(l*(T*ones-time))**j)/(np.math.factorial(j))
        out += (s2-s1)*f3
    return out


#E[F]
def gen_EF(s,number=20):
    L0 = s[0, :, :] * np.exp((drift_orig - 0.5 * volatility ** 2 - l * k) * (T))
    out = 0
    for j in range(number):
        ev = j * mu
        var = j * sigma ** 2 + (T ) * volatility ** 2
        arg2 = ((-ev) * torch.ones(batch_size, 1) + torch.log(F * torch.ones(batch_size, 1)) - torch.log(
            L0)) / np.sqrt(var)
        s2 = n.cdf(arg2)
        f3 = (np.exp(-l * (T )) * (l * (T)) ** j) / (np.math.factorial(j))
        out += (1 - s2) * f3
    return out


def gen_kappa_comp(s,number = 20):
    L = s * np.exp((drift_orig - 0.5 * volatility ** 2 - l * k) * (T * ones - time))
    out = 0
    for j in range(number):
        ev = j * mu * ones
        var = ones * j * sigma ** 2 + (T * ones - time) * volatility ** 2
        ev2 = (j+1)*mu*ones
        var2 = ones * (j+1) * sigma ** 2 + (T * ones - time) * volatility ** 2
        arg1 = (torch.log(F * ones) - torch.log(L) - ev) / torch.sqrt(var)
        s1 = n.cdf(arg1)
        arg2 = (torch.log(F * ones) - torch.log(L) - ev2) / torch.sqrt(var2)
        s2 = n.cdf(arg2)
        f3 = (torch.exp(-l * (T * ones - time)) * (l * (T * ones - time)) ** j) / (np.math.factorial(j))
        out += (s1-s2) * f3
    return l* out


w = torch.randn(sequence_len, batch_size,1)

tj = torch.zeros(sequence_len, batch_size,1)


for bn in range(batch_size):
    cum_time = np.random.exponential(1 / rates[0])
    while (cum_time < T):
        indx = int(cum_time / dt)

        tj[indx, bn] += 1

        cum_time += np.random.exponential(1 / rates[0])


ln = torch.distributions.log_normal.LogNormal(mu*ones,sigma*ones)
y = ln.sample()
gamma = y-ones
gamma1 = G*gamma
gamma2 = (ones)/(ones+gamma1) - ones

def gen_s(gamma,tj,w):
    s = ones*s0
    for i in range(sequence_len-1):
        ds = s[i,:,:]* drift* dt + s[i,:,:] * volatility * sqrdt * w[i,:,:] + s[i,:,:] *gamma[i,:,:]* tj[i,:,:]
        s[i+1,:,:] = s[i,:,:] + ds
    return s

s = gen_s(gamma,tj,w)

beta = gen_beta(s)

kappa = gen_kappa(s,y)

kappa_comp = gen_kappa_comp(s)

EF = gen_beta(s)[0,:,:]/volatility - F*gen_EF(s)

def gen_F(beta, kappa, EF,kappa_comp):
    f = EF.unsqueeze(0).repeat(sequence_len,1,1)
    for i in range(sequence_len-1):
        df = beta[i,:,:]*w[i,:,:]*sqrdt + kappa[i,:,:]*tj[i,:,:] - kappa_comp[i,:,:]*dt
        f[i+1,:,:] = f[i,:,:] + df

    return f


f = gen_F(beta,kappa,EF,kappa_comp)

fT = f[-1,:,:]

sT = s[-1,:,:]
Ftmp = torch.ones(batch_size, dim) * F
fT2 = torch.max(torch.zeros(batch_size,dim), sT-Ftmp)
torch.mean(fT2)

err = fT-fT2

i = np.random.randint(0,batch_size)
plt.plot(t1,f[:,i,0])
plt.plot(t1,s[:,i,0])
plt.axhline(y=fT2[i,0])
plt.show()



def plot(x,i):
    if i==0:
        i = np.random.randint(batch_size)
    plt.plot(t1,x[:,i,0])
    plt.show()
    return None


def plot2(x,y,i):
    if i==0:
        i = np.random.randint(batch_size)
    plt.plot(t1,x[:,i,0],color="red")
    plt.plot(t1, y[:, i, 0], color="blue")
    plt.show()
    return None


def EF_MC(n):
    bm = np.sqrt(T)*torch.randn(n)
    poisson = np.random.poisson(T*l, n)
    tjs = []
    for i in poisson:
        transformeds = np.random.normal(mu,sigma,i)
        tjs.append(np.sum(transformeds))
    tjs = torch.tensor(tjs)

    s = s0*torch.exp(((drift_orig - k* l) - 0.5*volatility**2)*T*torch.ones(n) + volatility*bm + tjs)

    ftmp = torch.ones(n) * F
    f = torch.max(torch.zeros(n), s - ftmp)

    op_val = torch.mean(f)
    return op_val









