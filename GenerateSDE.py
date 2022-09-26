import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson

#Generates an SDE of the form dX_t = X_t[a(t)dt + b(t)dB_t + c(t,z)\tilde{N}(dz,dt)]
#Constant coefitients

def genSDE(initial,control):
    a=0.1
    b=0.3
    l=10
    s0 = initial
    T = 1
    dt = 0.001
    n = int(T / dt)
    t = np.linspace(0., T, n)
    out = control
    y0 = 0.5

    sqrtdt = np.sqrt(dt)
    s = np.zeros(n)
    s[0] = s0
    y = np.zeros(n)
    y[0]=y0
    #np.random.seed(1)
    bg = np.random.normal(0,1,n)
    #np.random.seed(1)

    N = np.random.poisson(l*T)
    #np.random.seed(2)

    uni = np.sort(np.random.uniform(0,1,N))
    pois = 0
    j=0
    p = np.zeros(n)
    p[0] = 0


    for i in range(n - 1):
        if j < N-1 and pois == 0 and i*dt >= uni[j]:
            j += 1
            #np.random.seed(i)

            pois = 1 - (2*np.random.randint(2))
        else:
            pois = 0
        s[i + 1] = s[i] + out*(a * s[i] * dt + b*s[i] * sqrtdt * bg[i] + 0.2*s[i]*pois) #- 0.1*s[i]*l*dt
        y[i+1] = y[i] + y[i]*(a*dt + b * sqrtdt*bg[i] + 0.2*pois)
    return t, s, y


t,s, y = genSDE()

plt.plot(t,s)
plt.plot(t,y,"red")
plt.show()
print(s[-1])


K = 0.6
loss = 0
n = 0
for i in range(1000):
    t,s,y = genSDE(0.1,1.5)
    if np.min(s) <= 0:
        continue

    F = max(0,y[-1]-K)
    l = 0.5*(s[-1]-F)**2
    loss +=l
    n += 1


print(loss/n)










##################################ALTERNATIVE GENERATOR
a = 0.1
b= 0.1
c = 0.1

s0 = 1
T = 1
dt = 0.01
n = int(T / dt)
t = np.linspace(0., T, n)

sqrtdt = np.sqrt(dt)
s = np.zeros(n)
s[0] = s0
bg = np.random.normal(0, 1, n)
N = np.random.poisson(l * T)
uni = np.sort(np.random.uniform(0, 1, N))

tj = n*[0]
for u in uni:
    indx = int(u/dt)
    jumpsize = 1 - (2*np.random.randint(2)) #enakomerno -1,1
    tj[indx] += jumpsize

for i in range(n-1):
    s[i+1] = s[i] + a* s[i] * dt + b* s[i] *sqrtdt* bg[i] + 2*c * s[i] * tj[i]

plt.plot(t,s)
plt.show()

sl = 30
bs = 1000
dim = 5
dt = 1/sl

tj = torch.zeros(sl,bs,dim)





for bn in range(bs):
    for dn in range(dim):
        cum_time = np.random.exponential(1/rates[dn])
        while(cum_time < T):
            indx = int(cum_time/dt)
            jumpsize = 1 - (2 * np.random.randint(2))  # enakomerno -1,1
            tj[indx,bn, dn] += jumpsize

            cum_time += np.random.exponential(1/rates[dn])










def init_brownian():
    return torch.randn(sequence_len, batch_size, dimension)


w = init_brownian()


output_seq = torch.empty((sequence_len,
                          batch_size,
                          dimension))

input_seq = torch.empty((sequence_len,
                         batch_size,
                         dimension))

stock_seq = torch.empty((sequence_len,
                         batch_size,
                         dimension))

#xtmp = self.fc(input)
# x = self.activation(xtmp)
x = torch.ones(batch_size, dimension)*0.5
x0 = x

ones = torch.ones(batch_size, dimension)
# x0 = input
# x = input*0.2

# init the both layer cells with the zeroth hidden and zeroth cell states

s = torch.ones(batch_size, dimension) * 0
stock_seq[0] = s
input_seq[0] = x
# for every timestep use input x[t] to compute control out from hiden state h1 and derive the next imput x[t+1]
for t in range(sequence_len):
    # get the hidden and cell states from the first layer cell

    # out = self.activation(out)
    out = torch.rand(batch_size, dimension)
    output_seq[t] = out
    if t < sequence_len - 1:
        x = x + torch.mean(x, dim=0) * torch.mean(out, dim=0) * ones * dt + volatility * sqrdt * w[t]
        for i in range(batch_size):
            dimones = torch.ones(dimension)
            s[i,:] = dimones * (dt*t)**(i+1)
        input_seq[t + 1] = x
        stock_seq[t + 1] = s
# return the output and state sequence







def loss3(x,x0,out_seq):
    Ftmp = torch.ones(batch_size, dim) * F
    terminal = torch.mean(torch.square(torch.norm(x - Ftmp, dim=1)))
    start = torch.mean(torch.square(torch.norm(x0, dim=1)))
    integral = torch.trapz(torch.square(torch.norm(out_seq,dim=2)),dx= dt,dim=0)
    ongoing = torch.mean(integral)
    return start + ongoing + terminal






tst = torch.trapz(torch.square(torch.norm(stock_seq,dim=2)),dx= dt,dim=0)

ss = stock_seq.detach().cpu().numpy()

plt.plot(t1,ss[:,0,0],"palegreen")
plt.plot(t1,ss[:,1,0],"red")
plt.plot(t1,ss[:,2,0],"black")
plt.show()