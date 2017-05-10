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

T = 5
N = 5 # time steps occurrence
γ = 0.98 # discount factor
n_trials = 5
stimuli = [0, 2]

tdmodel = model.TDModel(λ=0, α=1, γ=γ, N=N, T=T, stimuli=stimuli)
for _ in range(n_trials):
    tdmodel.trial([True, True], reward=4)

fig = plt.figure()
ax = fig.gca(projection="3d")

X = np.arange(0, T, T/N)
Y = np.arange(0, n_trials)
X, Y = np.meshgrid(X, Y)
Z = tdmodel.δ_history

surface = ax.plot_surface(X, Y, Z, cmap=cm.magma, linewidth=0)

ax.set_zlim(-0.1, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surface, shrink=0.5, aspect=5)

#plt.savefig('figures/5_time_steps.png') #comment to subplot
plt.show()
plt.close()
