# ==============================================================================
"""Dopamine cell prediction activity"""
# ==============================================================================
__author__ = "Anais GRIMAL, Hoang-Yen PHAM"
__date__ = "03/04/2017"
# ==============================================================================

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from copy import *
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import *
import numpy as np

# ==============================================================================

T = 5
N = 23 # time steps occurrence
h = T/N # time step scale
γ = 0.98 # discount factor
α = 0.005 # learning rate
λ = 0.9 # eligibility trace parameter

# === Initialisation ===
k = 2 # stimuli occurrence. It has to be >= 1, even for no stimulus
#x = [[0 for t in range(N)] for i in range(k)] # state vectors of the stimuli, between 0.5 to 2 seconds
w = [[0 for t in range(N)] for i in range(k)] # weights vector per stimulus
#r = [0 for t in range(N)] # reward

pl = [[0 for t in range(N)] for i in range(k)] # Reward predictions
P = [0 for t in range(N)] # Total reward prediction
TD = [0 for t in range(N)] # Temporal difference
δ = [0.0 for t in range(N)] # Prediction error
e = [[0 for t in range(N)] for i in range(k)]# Eligibility trace
Δw = [[0.0 for t in range(N)] for i in range(k)] # Weight change

δ_plot = [] # to store δ's vectors
δ_temp = [0.0 for t in range(N)]
# === Trials ===
trials = 500
for j in range(trials):
    
    x = [[0 for t in range(N)] for i in range(k)]
    r = [0 for t in range(N)]
    e = [[0 for t in range(N)] for i in range(k)]
    s = [5, 15]
    r[20] = 1
    q = [0, 0]

    # action
    for t in range(N):
        # =0 at t=0, either way it isn't an N-sized list
        e[0] = np.multiply(e[0],λ)
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
            e[1] = np.multiply(e[1],λ)
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
   
        if t >= 1: # ==0 at t=0, either way it isn't an N-sized list
            TD[t] =  P[t-1] - γ * P[t] # <0 when predicts a reward at time step t+1
        
        δ[t] = r[t] - TD[t]
        if δ[t] > 1 : δ[t] = 1
        if δ[t] < -0.05 : δ[t] = -0.05
        
        Δw[0] = np.multiply(α * δ[t], e[0])
        w[0] = np.add(w[0], Δw[0])
        if k == 2:
            Δw[1] = np.multiply(α * δ[t], e[1])
            w[1] = np.add(w[1], Δw[1])

        if t == N-1 :
            e[0] = np.multiply(e[0],λ)
            e[0] = np.add(e[0],x[0])
            if k == 2:
                e[1] = np.multiply(e[1],λ)
                e[1] = np.add(e[1],x[1])
                
   # print(e[0])
    #print(e[1])
##    δ_temp = δ.copy()
##    δ_plot.append(δ_temp)
    #print(e)
    if j%10 == 0:
        δ_temp = δ.copy()
        δ_plot.append(δ_temp)

print(δ)

fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.arange(0,T,h)
Y = np.arange(0,trials,10)
X, Y = np.meshgrid(X, Y)

surface = ax.plot_surface(X, Y, δ_plot, cmap=cm.magma, linewidth=0)

ax.set_zlim(-0.05, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surface, shrink=0.5, aspect=5)
plt.show()
            
            
            
            
