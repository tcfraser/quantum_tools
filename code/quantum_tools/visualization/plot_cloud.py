import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
from ..config import *

def plot_3d_cloud(xyz_data, surfaces=[], labels=('x-label', 'y-label', 'z-label'), title='title'):
    xs, ys, zs = (xyz_data[:, i] for i in range(3))
    color_map = gaussian_kde(xyz_data.T)(xyz_data.T)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.hold(True) # For surfaces and scatter together
    ax.scatter(
        xs=xs,
        ys=ys,
        zs=zs,
        c=color_map,
        edgecolors='none',
    )
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(title)

    # plt.savefig(OUTPUT_DIR + 'test.pdf', format='pdf')
    plt.show()

def plot_3d_cloud_examples():
    num = 100
    x = np.random.normal(0,1,num)
    y = np.random.normal(0,1,num)
    z = np.random.normal(0,1,num)
    plot_3d_cloud(np.hstack((
        x[:,np.newaxis],
        y[:,np.newaxis],
        z[:,np.newaxis])))


if __name__ == '__main__':
    plot_3d_cloud_examples()

