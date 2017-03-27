# ==============================================================================
"""Dopamine cell prediction activity"""
# ==============================================================================
__author__ = "Anais GRIMAL, Hoang-Yen PHAM"
__version__ = "1.1"
__date__ = "20/03/2017"
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
N = 10 #time steps occurrence
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
P = pl[0] #total reward prediction
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
trials = 5
for j in range(trials):
    x = [[0 for t in range(N)] for i in range(k)]
    r = [0 for t in range(N)]

    x[0][5] = 1
    r[6] = 1
    q = [0 for i in range(k)]
    
    # action
    for t in range(N):
        for i in range(k):
            if x[i][t] == 1:
                q[i] = t
            
            pl[i][t] = x[i][t] * w[i][t] #reward predictions
            P[t] += pl[i][t]
            
            if t >= 1: # ==0 at t=0, either way it isn't an N-sized list
                e[i][t] = lamb*e[i][t-1] + x[i][t-1]
                TD[t] =  P[t-1] - gamma * P[t] # <0 when predicts a reward at time step t+1
            
            delta[t] = r[t] - TD[t]
            
            delta_w[i][t] = alpha * delta[t] * e[i][t]
            w[i][q[i]] += delta_w[i][t]

for i in range(N):
    if r[i] != 0: print("r", i*h, r[i])
    if w[0][i] != 0: print("w", i*h, w[0][i])
    if x[0][i] != 0: print("x", i*h, x[0][i])
    if e[0][i] != 0: print("e", i*h, e[0][i])
    if delta[i] != 0: print("prediction error", i*h,delta[i])

#test function to try it.


axisx = np.arange(0,T,h)
plt.plot(axisx, delta, 'k', linewidth=4)
plt.plot(axisx, r, 'ro', axisx, x[0], 'go')
# axes limits
axes = plt.gca()
axes.set_ylim([0, 3])
plt.show()
            
            
            
            
