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
N = 23 # time steps occurrence
γ = 0.98 # discount factor
n_trials = {'0.005':800, '0.05':400, '0.5':40, '1':40}
stimuli = [5, 15]
cpt = 0

#fig = plt.figure(figsize=plt.figaspect(1)) #uncomment to subplot

for λ in [1.0, 0.9, 0.6, 0.3, 0.0]:
    for α in [0.005, 0.05, 0.5, 1]:
        cpt += 1
        tdmodel = model.TDModel(λ=λ, α=α, γ=γ, N=N, T=T, stimuli=stimuli)
        for _ in range(n_trials[str(α)]):
            tdmodel.trial()

        fig = plt.figure() #comment to subplot
        ax = fig.gca(projection="3d") #comment to subplot
        
        #ax = fig.add_subplot(5, 4, cpt, projection='3d') #uncomment to subplot
        #fig.tight_layout() #uncomment to subplot
        
        X = np.arange(0, T, T/N)
        Y = np.arange(0, n_trials[str(α)])
        X, Y = np.meshgrid(X, Y)
        Z = tdmodel.δ_history

        surface = ax.plot_surface(X, Y, Z, cmap=cm.magma, linewidth=0)

        ax.set_zlim(-0.1, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        #fig.colorbar(surface, shrink=0.5, aspect=5)
        #plt.axis("off") #uncomment to subplot

        plt.savefig('figures/dopamine_cell_RL_3D%s.png' % (cpt)) #comment to subplot
        #plt.show()
        
#plt.close()
