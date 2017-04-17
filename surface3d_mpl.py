from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import IndexLocator, FormatStrFormatter

import numpy as np

def surface3d(X, Y, Z, show=True, x_label=None, y_label=None, z_label=None, title=''):

    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(IndexLocator(0.5, 0))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # White background
    for item in (ax.xaxis, ax.yaxis, ax.zaxis):
        item.set_pane_color((1.0, 1.0, 1.0, 1.0))
        item.label.set_size(7)

    for item in (ax.get_xticklabels(), ax.get_yticklabels(), ax.get_zticklabels()):
        plt.setp(item, fontsize=7)

    # Labels
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if z_label is not None:
        ax.set_zlabel(z_label)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.title(title)
    plt.show()
