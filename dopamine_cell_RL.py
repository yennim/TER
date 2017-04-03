# ==============================================================================
"""Dopamine cell prediction activity"""
# ==============================================================================
__author__ = "Anais GRIMAL, Hoang-Yen PHAM"
__date__ = "03/04/2017"
# ==============================================================================

from matplotlib import pyplot as plt
from random import *
import numpy as np

##from __future__ import division

##from bokeh.core.properties import Any, Dict, Instance, String
##from bokeh.io import show
##from bokeh.models import ColumnDataSource, Div, Column,  LayoutDOM

# ==============================================================================

T = 5
N = 25 #time steps occurrence
h = T/N #time step scale
gamma = 0.98 #discount factor
alpha = 0.05 #learning rate
lamb = 0 #eligibility trace parameter

# === Initialisation ===
k = 1 #stimuli occurrence >= 1, even for no stimulus
x = [[0 for t in range(N)] for i in range(k)] #state vectors of the stimuli, between 0.5 to 2 seconds
w = [[0 for t in range(N)] for i in range(k)] #weights vector per stimulus
r = [0 for t in range(N)] #reward

# === Reward prediction ===
pl = [[x[i][t]*w[i][t] for t in range(N)] for i in range(k)] #reward predictions
P = [0 for t in range(N)] #total reward prediction
if k > 1:
    for t in range(N):
        for i in range(1,k): #begin at 1 <-- P = pl[0]
            P[t] += pl[i][t]

# === Temporal difference ===
TD = [0]
for t in range(1,N):
    TD.append(P[t-1] - gamma*P[t])


# === Prediction error ===
delta = [(r[t] - TD[t]) for t in range(N)]


# === Eligibility trace ===
e = [[0] for i in range(k)]

for t in range(1,N):
    for i in range(k):
        e[i].append(lamb*e[i][t-1] + x[i][t-1])

# === Weight change ===
delta_w = [[alpha * delta[t] * e[i][t] for t in range(N)] for i in range(k)] #the last time step is ignored

for t in range(N):
    for i in range(k):
        w[i][t] +=  delta_w[i][t] #weights update    

# === Trials ===
trials = 100
for j in range(trials):
    x = [[0 for t in range(N)] for i in range(k)]
    r = [0 for t in range(N)]

    s = [5]
    r[20] = 1
    q = 0

    # action
    for t in range(N):
        for i in range(k):
            if t >= 1: # ==0 at t=0, either way it isn't an N-sized list
                e[i] = np.multiply(e[i],lamb)
                e[i] = np.add(e[i],x[i])

            if t == s[0]:
                x[0][q] = 1
                
            if 1 in x[i] and t > s[0]:
                try :
                    x[i][q] = 0
                    x[i][q+1] = 1
                    q += 1
                except: pass
            
            pl[i] = np.multiply(x[i],w[i]) #reward predictions
            P[t] = np.sum(pl[i])
            
            if t >= 1: # ==0 at t=0, either way it isn't an N-sized list
                TD[t] =  P[t-1] - gamma * P[t] # <0 when predicts a reward at time step t+1
            
            delta[t] = r[t] - TD[t]
            delta_w[i] = np.multiply(alpha * delta[t], e[i])
            w[i] = np.add(w[i], delta_w[i])



##print("ce qu'on veut 1:", (alpha**2)*gamma)
##print("ce qu'on veut 2:", (2*alpha)-(alpha**2))

for i in range(N): #print des valeurs
    if r[i] != 0: print("r", i*h, r[i])
    if w[0][i] != 0:
        print("w0", i*h, w[0][i])
    if k>1 and w[1][i] != 0: print("w1", i*h, w[1][i])
    if x[0][i] != 0: print("x0", i*h, x[0][i])
    if k>1 and x[1][i] != 0: print("x1", i*h, x[1][i])
    if e[0][i] != 0: print("e0", i*h, e[0][i])
    if k>1 and e[1][i] != 0: print("e1", i*h, e[1][i])
    if delta[i] != 0: print("prediction error", i*h,delta[i])

#test function to try it.

axisx = np.arange(0,T,h)
plt.plot(axisx, delta, 'k', linewidth=4)
plt.plot(s[0]*h, 1, 'go')
plt.plot(axisx, r, 'ro')
if k > 1:
    plt.plot(s[1]*h, 1, 'go')
# axes limits
axes = plt.gca()
axes.set_ylim([0, 1.5])
plt.show()
            
            
            
            
