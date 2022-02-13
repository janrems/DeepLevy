import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson

#Generates an SDE of the form dX_t = X_t[a(t)dt + b(t)dB_t + c(t,z)\tilde{N}(dz,dt)]
#Constant coefitients

def genSDE(a=0.1,b=0.05,l = 10):
    s0 = 1
    T = 1
    dt = 0.001
    n = int(T / dt)
    t = np.linspace(0., T, n)

    sqrtdt = np.sqrt(dt)
    s = np.zeros(n)
    s[0] = s0
    bg = np.random.normal(0,1,n)
    N = np.random.poisson(l*T)
    uni = np.sort(np.random.uniform(0,1,N))
    pois = 0
    j=0
    p = np.zeros(n)
    p[0] = 0

    for i in range(n - 1):
        if j < N-1 and pois == 0 and i*dt >= uni[j]:
            j += 1
            pois = 1 - (2*np.random.randint(2))
        else:
            pois = 0
        s[i + 1] = s[i] + a * s[i] * dt + b*s[i] * sqrtdt * bg[i] - 0.1*s[i]*pois #- 0.1*s[i]*l*dt
        p[i+1] = p[i] + 1*pois  - l*dt
    return t, s, p


t,s, p = genSDE()

plt.plot(t,s)
plt.show()











































