# ==============================================================================
"""Dopamine cell prediction activity"""
# ==============================================================================
__author__ = "Anais GRIMAL, Hoang-Yen PHAM"
__date__ = "03/04/2017"
# ==============================================================================

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np

import model
# ==============================================================================

n_trials = 800

T = 5
N = 23 # time steps occurrence
γ = 0.98 # discount factor
λ = 0 # eligibility trace parameter
α = 0.005 # learning rate

# running the model
tdmodel = model.TDModel(λ=λ, α=α, γ=γ, N=N, T=T)
for _ in range(n_trials):
    tdmodel.trial()

# plots
print(tdmodel.δ_history[-1])

fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.arange(0, T, T/N)
Y = np.arange(0, n_trials)
X, Y = np.meshgrid(X, Y)

surface = ax.plot_surface(X, Y, tdmodel.δ_history, cmap=cm.magma, linewidth=0)

ax.set_zlim(-0.05, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surface, shrink=0.5, aspect=5)

plt.savefig('figures/dopamine_cell_RL_3D.pdf')
plt.show()
