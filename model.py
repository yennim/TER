"""
Model implementation
"""

import numpy as np


class TDModel:

    def __init__(self, λ=0.9,             # eligibility trace parameter
                       α=0.005,           # learning rate
                       γ=0.98,            # discount factor
                       N=23,              # number of time steps
                       T=5,
                       stimuli=[5, 15]):  # trial duration
        self.λ, self.α, self.γ = λ, α, γ
        self.N, self.T = N, T

        self.k = len(stimuli)  # stimuli occurrence. It has to be >= 1, even for no stimulus
        self.stimuli  = stimuli

        self.w = np.zeros((self.k, self.N))  # weights vector per stimulus
        self.δ_history = []  # to store the history of δ


    def compute_task(self, stim=(True, True), reward=20):
        """Compute stimuli and reward vectors."""
        # stimuli affectation
        x = np.zeros((self.k, self.N, self.N))
        for (x_l, s_t, s_present) in zip(x, self.stimuli, stim):
            if s_present:
                for t in range(self.N):
                    if s_t <= t+1:
                        x_l[t][t-s_t] = 1
        # rewards affectation
        r         = np.zeros(self.N)
        r[reward] = 1  # reward

        return x, r


    def trial(self, stim=(True, True), reward=20):
        """Compute a trial"""
        ones = np.ones(self.N)

        P    = np.zeros(self.N)            # total reward prediction
        δ    = np.zeros(self.N)            # prediction error
        e    = np.zeros((self.k, self.N))  # eligibility trace vector

        x, r = self.compute_task(stim=stim, reward=reward)

        # actual trial
        for t in range(self.N):
            for l in range(self.k):
                P_l = np.dot(x[l][t], self.w[l])
                P[t] += P_l

            if t > 0:  # TD[t] == 0 at t = 0 : P[t-1] doesn't exist
                TD_t = P[t-1] - self.γ * P[t]  # <0 when predicts a reward at time step t+1
            else:
                TD_t = 0

            δ[t] = r[t] - TD_t
            δ[t] = min(1.0, max(-0.05, δ[t]))

            for l in range(self.k):
                if t > 0:  # t=0 : x[i][t-1] doesn't exist
                    if self.λ == 1:  # case λ=1 isolated...
                        e[l] = self.γ * self.λ * (e[l] + ones)
                    else:
                        e[l] = self.λ * e[l] + x[l][t-1]

                Δw_l_t = self.α * δ[t] * e[l]
                self.w[l] += Δw_l_t

        self.δ_history.append(δ.copy())
        return e
