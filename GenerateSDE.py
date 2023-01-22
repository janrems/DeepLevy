import numpy as np
import matplotlib.pyplot as plt


#Parametri particije
T = 1
sequence_len = 10000000
dt = T/sequence_len
t=np.arange(0,T,dt)
sqrdt = np.sqrt(dt)

#Generiramo standardno normalne s.s., ki nam dajo Brownovo gibanje
nor = np.random.normal(0,1,sequence_len)

#Skonstruiramo Brownovo gibanje
bg = np.zeros(sequence_len)
for i in range(sequence_len - 1):
    bg[i + 1] = bg[i] + sqrdt * nor[i]

#Graf Brownovega gibanja
plt.plot(t, bg)
plt.show()



#Geometricno brovnovo gibanje dS_t = S_t(mu dt + sigma dBt)
mu = 0.3
sigma = 0.2
s = np.zeros(sequence_len)
s[0] = 1
for i in range(sequence_len-1):
    s[i+1] = s[i] + mu*s[i]*dt + sigma*s[i]*(bg[i+1] - bg[i])


#Analiticna resitev gornje SDE
s2 = 1*np.exp((mu-0.5*sigma**2)*t + sigma*bg)

#Primerjava rezultatov
plt.plot(t,s)
plt.plot(t,s2,color="r")
plt.show()





#Brownov most

#Preko formule X_t = (1-t)M_t za dM_t = 1/(1-t)dBt
m = np.zeros(sequence_len)
m[0] = 0

#Preko SDE dX_t = -X_t/(1-t)dt + dBt
x2 = np.zeros(sequence_len)
x2[0] = 0
for i in range(sequence_len-1):
    m[i+1] = m[i] + (sqrdt*nor[i])/(1-t[i])
    x2[i+1] = x2[i] -x2[i]/(1-t[i])*dt + sqrdt*nor[i]
x = (1-t)*m


#Alternativni zapis za Brownov most
y = bg - t*bg[-1]

#Primerjava
#Pozor: grafa za x in x2 se za grobo diskretizacijo (majhen sequence_len) razlikujeta od grafa y.
# Razlika se povečuje s pretečenim časom

plt.plot(t,x)
plt.plot(t,y, color="black")
plt.plot(t,x2, color= "g")
plt.axhline(y=0,color="r")
plt.show()

