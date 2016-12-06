import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import product
import numpy as np

def triangle_plot(pd, title=None, subtitle=None):
    tall_support = pd._support.reshape((16,4))
    plt.figure()
    im = plt.imshow(tall_support, cmap='gnuplot2_r', interpolation='none')
    plt.title('{}\n{}'.format(title, subtitle))
    plt.xticks(range(tall_support.shape[1]), ['C = {}'.format(i) for i in range(4)], rotation='vertical')
    plt.yticks(range(tall_support.shape[0]), ['A = {}, B = {}'.format(i, j) for i, j in product(range(4), range(4))], rotation='horizontal')
    plt.colorbar(im, orientation='horizontal')
    plt.show()

def triangle_plot_2(pd, title=None, save=None, max_val=None, brazil_mods=False, bits=False):
    shaped_support = pd._support.reshape((4,4,4))
    if brazil_mods:
        shaped_support = pd._support.reshape((4,4,4))[:,:,np.array([1,3,0,2])][np.array([3,2,0,1]),:,:] # Hack for Brazil
    fig, axarr = plt.subplots(1, 4)
    # fig.tight_layout()
    max_val = max_val if max_val is not None else shaped_support.max()
    ticks = ['{}'.format(i) for i in range(4)]
    if bits:
        ticks = ['00', '01', '10', '11']

    for i in range(4):
        im = axarr[i].imshow(shaped_support[:,:,i], vmin=0, vmax=max_val, cmap='hot_r', interpolation='none')
        axarr[i].set_xlabel('B')
        axarr[i].set_title('C = {}'.format(ticks[i]))
        axarr[i].set_xticks(range(4))
        axarr[i].set_xticklabels(ticks)
        axarr[i].tick_params(axis='x', length=0)
        axarr[i].set_yticks([])
        axarr[i].set_yticks([t-0.5 for t in range(4)], minor=True)
        axarr[i].set_xticks([t-0.5 for t in range(4)], minor=True)
        axarr[i].grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
        axarr[i].set_yticklabels([])
        axarr[i].tick_params(axis='y', length=0)
    axarr[0].set_ylabel('A', rotation='horizontal', labelpad=15)
    axarr[0].set_yticks(range(4))
    axarr[0].set_yticklabels(ticks)
    axarr[0].tick_params(axis='y', length=0)
    fig.subplots_adjust(wspace=0.1, top=1)
    cax, kw = mpl.colorbar.make_axes([ax for ax in axarr.flat], location='bottom')
    plt.colorbar(im, cax=cax, **kw)

    if title is not None:
        fig.set_size_inches(8, 3.5, forward=True)
        plt.suptitle(title, fontsize=20)
    else:
        fig.set_size_inches(8, 3, forward=True)
    if save is not None:
        plt.savefig(save, format='pdf')
    plt.show()


def triangle_plot_small(pd, title=None, save=None, max_val=None):
    shaped_support = pd._support.reshape((2,2,2))

    fig, axarr = plt.subplots(1, 2)
    # fig.tight_layout()
    max_val = max_val if max_val is not None else shaped_support.max()
    ticks = ['{}'.format(i) for i in range(2)]

    for i in range(2):
        im = axarr[i].imshow(shaped_support[:,:,i], vmin=0, vmax=max_val, cmap='hot_r', interpolation='none')
        axarr[i].set_xlabel('B')
        axarr[i].set_title('C = {}'.format(i))
        axarr[i].set_xticks(range(2))
        axarr[i].set_xticklabels(ticks)
        axarr[i].tick_params(axis='x', length=0)
        axarr[i].set_yticks([])
        axarr[i].set_yticks([t-0.5 for t in range(2)], minor=True)
        axarr[i].set_xticks([t-0.5 for t in range(2)], minor=True)
        axarr[i].grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
        axarr[i].set_yticklabels([])
        axarr[i].tick_params(axis='y', length=0)
    axarr[0].set_ylabel('A', rotation='horizontal', labelpad=15)
    axarr[0].set_yticks(range(2))
    axarr[0].set_yticklabels(ticks)
    axarr[0].tick_params(axis='y', length=0)
    fig.subplots_adjust(wspace=0.1)
    # cax, kw = mpl.colorbar.make_axes([ax for ax in axarr.flat], location='bottom')
    # plt.colorbar(im, cax=cax, **kw)

    plt.suptitle(title, fontsize=20)
    fig.set_size_inches(4, 3, forward=True)
    if save is not None:
        plt.savefig(save, format='pdf')
    plt.show()