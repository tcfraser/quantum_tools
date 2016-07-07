import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

def plot_matrix(m):
    if m.nnz < 1e5:
        m = coo_matrix(m)
        fig = plt.figure()
        ax = fig.add_subplot(111, axisbg='white')
        ax.scatter(m.col, m.row, marker='s', c=m.data)
        # ax.spy(m, markersize=4)
        padding = 0.5
        ax.set_xlim(0-padding, m.shape[1]+padding)
        ax.set_ylim(0-padding, m.shape[0]+padding)
        # ax.set_aspect('equal')
        # for spine in ax.spines.values():
        #     spine.set_visible(False)
        ax.invert_yaxis()
        # ax.set_xticks([])
        # ax.set_yticks([])
        plt.show()
    else:
        print("m is too big to plot.")
