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
k = 2 #stimuli occurrence >= 1, even for no stimulus
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

# === Learning function ===
def TD_RL(i, x, r, N, e, pl, P, delta, delta_w, w):
    q = 0
    
    # action
    for t in range(N):
        if x[t] == 1:
            q = t
        
        pl[i][t] = x[t] * w[i][t] #reward predictions
        P[t] += pl[i][t]
        
        if t >= 1: # ==0 at t=0, either way it isn't an N-sized list
            e[i][t] = lamb*e[i][t-1] + x[t-1]
            TD[t] =  P[t-1] - gamma * P[t] # <0 when predicts a reward at time step t+1
        
        delta[t] = r[t] - TD[t]
        
        delta_w[i][t] = alpha * delta[t] * e[i][t]
        w[i][q] += delta_w[i][t]
    

# === Trials ===
trials = 10
for j in range(trials):
    x = [[0 for t in range(N)] for i in range(k)]
    r = [0 for t in range(N)]

    x[0][4] = 1
    x[1][5] = 1
    r[6] = 1
    q = [0 for i in range(k)]

##    TD_RL(0, x[0], x[1], N, e, pl, P, delta, delta_w, w)
##    TD_RL(1, x[1], r, N, e, pl, P, delta, delta_w, w)
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
    
##            if i == 0:
##                print("eligibilité", e[i][t])
##                print("delta", delta[t])
##                print("delta_w",delta_w[i][t])
##                print("recompense", r[t] == 1)
##                print("w0", w[i][q[i]])

for i in range(N):
    if r[i] != 0: print("r", i*h, r[i])
    if w[0][i] != 0: print("w0", i*h, w[0][i])
    if k>1 and w[1][i] != 0: print("w1", i*h, w[1][i])
    if x[0][i] != 0: print("x0", i*h, x[0][i])
    if k>1 and x[1][i] != 0: print("x1", i*h, x[1][i])
    if e[0][i] != 0: print("e0", i*h, e[0][i])
    if k>1 and e[1][i] != 0: print("e1", i*h, e[1][i])
    if delta[i] != 0: print("prediction error", i*h,delta[i])

#test function to try it.

axisx = np.arange(0,T,h)
plt.plot(axisx, delta, 'k', linewidth=4)
plt.plot(axisx, r, 'ro')
plt.plot(axisx, x[0], 'go')
if k > 1:
    plt.plot(axisx, x[1], 'go')
# axes limits
axes = plt.gca()
axes.set_ylim([0, 3])
plt.show()
            
            
            
            
