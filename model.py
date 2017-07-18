"""
Model implementation
"""

import numpy as np


class TDModel:

    def __init__(self, λ=0.9,             # eligibility trace parameter
                       α=0.005,           # learning rate
                       γ=0.98,            # discount factor
                       N=23,              # number of time steps
                       T=5,               # trial duration
                       stimuli=[5, 15]):  # stimuli's time steps
        self.λ, self.α, self.γ = λ, α, γ
        self.N, self.T = N, T

        self.k = len(stimuli)  # stimuli occurrence. It has to be >= 1, even for no stimulus
        self.stimuli  = stimuli

        self.w = np.zeros((self.k, self.N))  # weights vector per stimulus
        self.δ_history = []  # store the history of δ


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

        P    = np.zeros(self.N)            # total reward prediction
        δ    = np.zeros(self.N)            # prediction error
        e    = np.zeros((self.k, self.N))  # eligibility trace vector

        x, r = self.compute_task(stim=stim, reward=reward)

        # actual trial
        for t in range(self.N):
            for l in range(self.k):
                P_l = np.dot(x[l][t], self.w[l]) # reward prediction per stimulus
                P[t] += P_l

            if t > 0:   
                TD_t = P[t-1] - self.γ * P[t]
            else:
                TD_t = 0

            δ[t] = r[t] - TD_t
            δ[t] = min(1.0, max(-0.05, δ[t]))   # prediction error limited from -0.05 to 1

            for l in range(self.k):
                if t > 0:
                    e[l] = self.λ * e[l] + x[l][t-1]

                Δw_l_t = self.α * δ[t] * e[l]
                self.w[l] += Δw_l_t

        self.δ_history.append(δ.copy())
