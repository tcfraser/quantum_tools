import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

def plot_coo_matrix(m):
    if m.nnz < 100000:
        if not isinstance(m, coo_matrix):
            m = coo_matrix(m)
        fig = plt.figure()
        ax = fig.add_subplot(111, axisbg='white')
        # ax.plot(m.col, m.row, 's', color='black', ms=3)
        ax.spy(m, markersize=4)
        # ax.set_xlim(0, m.shape[1])
        # ax.set_ylim(0, m.shape[0])
        # ax.set_aspect('equal')
        # for spine in ax.spines.values():
        #     spine.set_visible(False)
        # ax.invert_yaxis()
        # ax.set_aspect('equal')
        # ax.set_xticks([])
        # ax.set_yticks([])
        plt.show()
    else:
        pass