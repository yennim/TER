"""
Model implementation
"""

import numpy as np


class TDModel:

    def __init__(self, λ = 0.9,   # eligibility trace parameter
                       α = 0.005, # learning rate
                       γ = 0.98,  # discount factor
                       N = 23,    # number of time steps
                       T = 5):    # trial duration
        self.λ, self.α, self.γ = λ, α, γ
        self.N, self.T = N, T

        self.k = 2 # stimuli occurrence. It has to be >= 1, even for no stimulus
        self.stimuli  = [5, 15]
        
        self.w = np.array([[0.0 for t in range(self.N)] for i in range(self.k)]) # weights vector per stimulus
        self.δ_history = [] # to store the history of δ

    def trial(self, stim1=True, stim2=True):
        P_l = np.array([[0.0 for t in range(self.N)] for i in range(self.k)]) # Reward predictions 
        P   = [ 0.0 for t in range(self.N)] # Total reward prediction 
        TD  = [ 0.0 for t in range(self.N)] # Temporal difference 
        δ   = [ 0.0 for t in range(self.N)] # Prediction error 
        Δw  = np.array([[0.0 for t in range(self.N)] for i in range(self.k)]) # Weight change 

        x   = np.array([[0.0 for t in range(self.N)] for i in range(self.k)])# Stimuli state vectors
        r   = [ 0.0 for t in range(self.N)] # rewards vector
        e   = np.array([[0.0 for t in range(self.N)] for i in range(self.k)]) # Eligibility trace vector
        s   = [] # stimulus time

        # stimuli affectation
        if stim1:
            self.k = 1
            s.append(self.stimuli[0])
            if stim2:
                self.k = 2
                s.append(self.stimuli[1])
        r[20] = 1 # reward
        q = [0, 0]

        # run
        for t in range(self.N):
            
            e[0] = self.λ * e[0] + x[0]

            if t == s[0]:
                x[0][q[0]] = 1

            if 1 in x[0] and t > s[0]:
                try:
                    x[0][q[0]] = 0
                    x[0][q[0]+1] = 1
                    q[0] += 1
                except: pass

            P_l[0] = x[0] * self.w[0]
            P[t] = np.sum(P_l[0]) 

            if self.k == 2: # second stimulus
                e[1] = self.λ * e[1] + x[1]

                if t == s[1]:
                    x[1][q[1]] = 1

                if 1 in x[1] and t > s[1]:
                    try:
                        x[1][q[1]] = 0
                        x[1][q[1]+1] = 1
                        q[1] += 1
                    except: pass

                P_l[1] = x[1] * self.w[1]
                P[t] += np.sum(P_l[1])

            if t >= 1: # ==0 at t=0, either way it isn't an N-sized list
                TD[t] =  P[t-1] - self.γ * P[t] # <0 when predicts a reward at time step t+1

            δ[t] = r[t] - TD[t]
            δ[t] = min(1.0, max(-0.05, δ[t]))

            Δw[0] = self.α * δ[t] * e[0]
            self.w[0] = self.w[0] + Δw[0]
            if self.k == 2: 
                Δw[1] = self.α * δ[t] * e[1]
                self.w[1] = self.w[1] + Δw[1] 

            if t == self.N - 1 :
                e[0] = self.λ * e[0] + x[0]
##                e[0] = np.multiply(e[0], self.λ) 
##                e[0] = np.add(e[0], x[0]) 
                if self.k == 2:
                    e[1] = self.λ * e[1] + x[1]
##                    e[1] = np.multiply(e[1], self.λ) 
##                    e[1] = np.add(e[1],x[1])

        self.δ_history.append(δ.copy())
