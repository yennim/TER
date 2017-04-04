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
N = 21 #time steps occurrence
h = T/N #time step scale
gamma = 0.98 #discount factor
alpha = 0.05 #learning rate
lamb = 0 #eligibility trace parameter

# === Initialisation ===
k = 2 #stimuli occurrence. It has to be >= 1, even for no stimulus
x = [[0 for t in range(N)] for i in range(k)] #state vectors of the stimuli, between 0.5 to 2 seconds
w = [[0 for t in range(N)] for i in range(k)] #weights vector per stimulus
r = [0 for t in range(N)] #reward

# === Reward prediction ===
pl = [[0 for t in range(N)] for i in range(k)] #reward predictions
P = [0 for t in range(N)] #total reward prediction

# === Temporal difference ===
TD = [0 for t in range(N)]

# === Prediction error ===
delta = [0 for t in range(N)]

# === Eligibility trace ===
e = [[0 for t in range(N)] for i in range(k)]

# === Weight change ===
delta_w = [[0 for t in range(N)] for i in range(k)] #the last time step is ignored

# === Trials ===
trials = 100
for j in range(trials):
    x = [[0 for t in range(N)] for i in range(k)]
    r = [0 for t in range(N)]
    e = [[0 for t in range(N)] for i in range(k)]

    s = [5, 15]
    r[20] = 1
    q = [0, 0]

    # action
    for t in range(N):
        if t >= 1: # ==0 at t=0, either way it isn't an N-sized list
            e[0] = np.multiply(e[0],lamb)
            e[0] = np.add(e[0],x[0])

        if t == s[0]:
            x[0][q[0]] = 1
        if 1 in x[0] and t > s[0]:
            try:
                x[0][q[0]] = 0
                x[0][q[0]+1] = 1
                q[0] += 1
            except: pass
        pl[0] = np.multiply(x[0],w[0]) #reward predictions
        P[t] = np.sum(pl[0])
        
        if k == 2:
            e[1] = np.multiply(e[1],lamb)
            e[1] = np.add(e[1],x[1])
            if t == s[1]:
                x[1][q[1]] = 1        
            if 1 in x[1] and t > s[1]:
                try:
                    x[1][q[1]] = 0
                    x[1][q[1]+1] = 1
                    q[1] += 1
                except: pass
            pl[1] = np.multiply(x[1],w[1])
            P[t] += np.sum(pl[1])
            
##        print("x%s=%s" % (t, x[0]))
##        print("y%s=%s" % (t, x[1]))
##        print("ex%s=%s" % (t, e[0]))
##        print("ey%s=%s" % (t, e[1]))
##        print("px%s=%s" % (t, pl[0]))
##        print("py%s=%s" % (t, pl[1]))
##        print("P%s=%s" % (t, P[t]))
              
        if t >= 1: # ==0 at t=0, either way it isn't an N-sized list
            TD[t] =  P[t-1] - gamma * P[t] # <0 when predicts a reward at time step t+1
        
        delta[t] = r[t] - TD[t]
        delta_w[0] = np.multiply(alpha * delta[t], e[0])
        w[0] = np.add(w[0], delta_w[0])
        if k == 2:
            delta_w[1] = np.multiply(alpha * delta[t], e[1])
            w[1] = np.add(w[1], delta_w[1])
            
##        print("deltawx%s=%s" % (t, delta_w[0]))
##        print("deltawy%s=%s" % (t, delta_w[1]))
##        print("wx%s=%s" % (t, w[0]))
##        print("wy%s=%s" % (t, w[1]))
##        print("")
##print("ce qu'on veut 1:", 2*(alpha**2)*gamma)
##print("ce qu'on veut 2:", (2*alpha)-(2*(alpha**2)))
            
for i in range(N):
##    if r[i] != 0: print("r", i*h, r[i])
##    if w[0][i] != 0: print("w0", i*h, w[0][i])
##    if k>1 and w[1][i] != 0: print("w1", i*h, w[1][i])
##    if x[0][i] != 0: print("x0", i*h, x[0][i])
##    if k>1 and x[1][i] != 0: print("x1", i*h, x[1][i])
##    if e[0][i] != 0: print("e0", i*h, e[0][i])
##    if k>1 and e[1][i] != 0: print("e1", i*h, e[1][i])
    if delta[i] != 0: print("prediction error", i*h, delta[i])

#test function to try it.

axisx = np.arange(0,T,h)
plt.plot(axisx, delta, 'k', linewidth=4)
plt.plot(s[0]*h, 1, 'go')
plt.plot(axisx, r, 'ro')
if k > 1:
    plt.plot(s[1]*h, 1, 'go')
# axes limits
axes = plt.gca()
axes.set_ylim([0, 2])
plt.show()
            
            
            
            
