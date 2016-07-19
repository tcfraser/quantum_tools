import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.sparse import coo_matrix

def plot_hypergraph(hg):
    if hg.nnz <= 1e6:
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
        print("Hypergraph has too many entries > 1e6.")
        
def plot_transversals(fts):
    plot_hypergraph(fts)
        
def visualize_transversal_overlap(fts):
    plt.matshow((fts.T * fts).todense(), interpolation='none')
    
def dense_plot_matrix(hg):
    plt.matshow(hg.todense(), interpolation='none')
