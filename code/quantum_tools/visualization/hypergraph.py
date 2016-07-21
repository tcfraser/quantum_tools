import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.sparse import coo_matrix

def plot_hypergraph(hg):
    upper_bound = 2e6
    if hg.nnz <= upper_bound:
        hg= coo_matrix(hg)
        fig = plt.figure()
        ax = fig.add_subplot(111, axisbg='white')
        ax.scatter(hg.col, hg.row, marker='s', s=1, edgecolor=None, c=hg.data)
        padding = 0.5
        ax.set_xlim(0-padding, hg.shape[1]+padding)
        ax.set_ylim(0-padding, hg.shape[0]+padding)
        ax.invert_yaxis()
        plt.show()
    else:
        print("Hypergraph has too many entries > {}.".format(upper_bound))
        
def plot_transversals(fts):
    plot_hypergraph(fts)
        
def transversal_overlap(fts):
    return (fts.T * fts).todense()

def visualize_transversal_overlap(fts):
    plt.matshow(transversal_overlap(fts), interpolation='none')
    
def dense_plot_matrix(hg):
    plt.matshow(hg.todense(), interpolation='none')
