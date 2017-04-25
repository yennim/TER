"""
Model implementation
"""

import numpy as np


class TDModel:

    def __init__(self, λ=0.9,   # eligibility trace parameter
                       α=0.005, # learning rate
                       γ=0.98,  # discount factor
                       N=23,    # number of time steps
                       T=5,
                       stimuli=[5,15]):    # trial duration
        self.λ, self.α, self.γ = λ, α, γ
        self.N, self.T = N, T

        self.k = len(stimuli) # stimuli occurrence. It has to be >= 1, even for no stimulus
        self.stimuli  = stimuli

        self.w = np.array([[0.0 for t in range(self.N)] for i in range(self.k)]) # weights vector per stimulus
        self.δ_history = [] # to store the history of δ

    def trial(self, stim=[True, True], reward=20):
        P_l = [ 0.0 for i in range(self.k)] # Reward predictions 
        P   = [ 0.0 for t in range(self.N)] # Total reward prediction 
        TD  = [ 0.0 for t in range(self.N)] # Temporal difference 
        δ   = [ 0.0 for t in range(self.N)] # Prediction error 
        Δw  = np.array([[0.0 for t in range(self.N)] for i in range(self.k)]) # Weight change 

        r   = [ 0.0 for t in range(self.N)] # rewards vector
        x   = np.array([[[0.0 for t in range(self.N)] for _ in range(self.N)] for i in range(self.k)])# Stimuli state vectors
        e   = np.array([[0.0 for t in range(self.N)] for i in range(self.k)]) # Eligibility trace vector
        s   = [] # stimulus time

        # stimuli affectation
        for (x_l, s_t, s_present) in zip(x, self.stimuli, stim):
            if s_present:
                for t in range(self.N):
                    if s_t <= t:
                        x_l[t][t-s_t] = 1

        r[reward] = 1 # reward

        # run
        for t in range(self.N):
            for i in range(self.k):
                P_l[i] = np.dot(x[i][t], self.w[i])
                P[t] += P_l[i]
                if t > 0:
                    e[i] = self.λ * e[i] + x[i][t-1]

            if t > 0: # ==0 at t=0, either way it isn't an N-sized list
                TD[t] =  P[t-1] - self.γ * P[t] # <0 when predicts a reward at time step t+1

            δ[t] = r[t] - TD[t]
            δ[t] = min(1.0, max(-0.05, δ[t]))

            for i in range(self.k):
                Δw[i] = self.α * δ[t] * e[i]
                self.w[i] += Δw[i]

##            print("e",e)
##            print("w",self.w)
##            print("δ",δ)
##            print("P",P)
##            print("\n")
        self.δ_history.append(δ.copy())

        return e
