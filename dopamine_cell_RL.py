# ==============================================================================
"""Dopamine cell prediction activity"""
# ==============================================================================
__author__ = "Anais GRIMAL, Hoang-Yen PHAM"
__version__ = "1.0"
__date__ = "01/03/2017"
# ==============================================================================


from matplotlib import pyplot as plt

T = 5
h = 0.005 #time step scale
N = int(T//h) #time steps occurrence
gamma = 0.98 #discount factor
alpha = 0.5 #learning rate
lamb = 0.5 #eligibility trace parameter

# === Initialisation ===
k = 2 #stimuli occurrence
x = [[0 for t in range(N)] for i in range(k)] #state vectors of the stimuli
w = [[0 for t in range(N)] for i in range(k)] #weights vector per stimulus
r = [0 for t in range(N)] #reward

r[750] = 1 #reward at time step #750


# === Reward prediction ===
pl = [[x[i][t]*w[t] for t in range(N)] for i in range(k)] #reward predictions
P = pl[0] #total reward prediction
if k > 1:
    for t in range(N):
        for j in range(1,k): #begin at 1 <-- P = pl[0]
            P[t] += pl[j][t]


# === Temporal difference ===
TD = [(P[t-1] - gamma*P[t]) for t in range(1, N)]


# === Prediction error ===
delta = [(r[t] - TD[t]) for t in range(N-1)] #the last time step is ignored


# === Eligibility trace ===
e = [[0 for t in range(N)] for i in range(k)]

for i in range(k):
    e[i][0] = 0

for t in range(1, N):
    for i in range(k):
        e[i][t] = lamb*e[i][t-1] + x[i][t-1]

# === Weight change ===
delta_w = [[alpha * delta[t] * e[i][t] for t in range(N-1)] for i in range(k)] #the last time step is ignored

for t in range(1,N-1):
    for i in range(k)
        w[i][t] +=  delta_w[i][t] #weights update
