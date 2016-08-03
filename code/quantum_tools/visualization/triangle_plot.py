import matplotlib.pyplot as plt
from itertools import product

def triangle_plot(pd, title=None, subtitle=None):
    tall_support = pd._support.reshape((16,4))
    plt.figure()
    im = plt.imshow(tall_support, cmap='gnuplot2_r', interpolation='none')
    plt.title('{}\n{}'.format(title, subtitle))
    plt.xticks(range(tall_support.shape[1]), ['C = {}'.format(i) for i in range(4)], rotation='vertical')
    plt.yticks(range(tall_support.shape[0]), ['A = {}, B = {}'.format(i, j) for i, j in product(range(4), range(4))], rotation='horizontal')
    plt.colorbar(im, orientation='horizontal')
    plt.show()