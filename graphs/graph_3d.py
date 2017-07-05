import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def δ_3d(model, ax=None, show=True, figsize=(10, 6), filename=None):
    X = np.arange(0, model.T, model.T/model.N)
    Y = np.arange(0, len(model.δ_history))
    Z = model.δ_history

    return surface3d(X, Y, Z, ax=ax, show=show, figsize=figsize, filename=filename)


def surface3d(X, Y, Z, ax=None, show=True, figsize=(10, 8), filename=None):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection='3d')

    XX, YY = np.meshgrid(X, Y)

    ax.view_init(azim=-51, elev=35)
    surface = ax.plot_surface(XX, YY, Z, cmap=mpl.cm.Spectral_r, linewidth=0)

    for child in ax.get_children():
        if isinstance(child, mpl.spines.Spine):
            child.set_color((0.4, 0.4, 0.4, 0.2))

    for item in (ax.xaxis, ax.yaxis, ax.zaxis):
        item.get_major_formatter().set_useOffset(False)
        item.set_pane_color((1.0, 1.0, 1.0, 1.0))
        item._axinfo['grid']['linewidth'] = 0.4
        item._axinfo['grid']['color']     = (0.2, 0.2, 0.2, 0.2)
        item._axinfo['color']             = (0.4, 0.4, 0.4, 0.2)
        item._axinfo['tick']['color']     = (0.4, 0.4, 0.4, 1.0)
        item._axinfo['tick']['outward_factor'] = 0.0
        item._axinfo['axisline']['color'] = (0.4, 0.4, 0.4, 1.0)
        item._axinfo['label']['color']    = (0.4, 0.4, 0.4, 0.2)
        item.line.set_color((0.4, 0.4, 0.4, 1.0))

    ax.set_zlim(0.0, 1.10)
    ax.set_ylim(0, Y[-1]+1)
    ax.set_xlim(X[0], X[-1])
    #ax.xaxis.get_major_formatter().set_useOffset(False)
    #ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.set_major_locator(mpl.ticker.LinearLocator(5))
    ax.zaxis.set_major_locator(mpl.ticker.FixedLocator((0.0, 0.5, 1.0)))
    ax.zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.02f'))

    if filename is not None:
        plt.tight_layout()
        plt.savefig('figures/{}.pdf'.format(filename))

    if show:
        plt.show()
    else:
        return ax
