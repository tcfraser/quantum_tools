# import matplotlib as mpl
# mpl.use("pgf")
# pgf_with_rc_fonts = {
#     "pgf.texsystem": "pdflatex"
# }
# mpl.rcParams.update(pgf_with_rc_fonts)

from ..contexts.quantum_caller import *
from ..statistics import *
from ..examples.symbolic_contexts import *
from ..examples.prob_dists import *
from ..rmt.unitary_param import *
from ..rmt.utils import *
from ..utilities import utils
from ..statistics.probability import *
from ..visualization.transversal_inequalities import *
from ..config import *
from ..inflation import marginal_equality
from itertools import permutations
import numpy as np
from scipy import optimize
from quantum_tools.visualization.triangle_plot import *
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

result_backlog = []
rvc = RandomVariableCollection.new(('A', 'B'), (2,2))

def minimize_callback(f):
    logged_results = []
    print(len(result_backlog))
    result_backlog.append(logged_results)
    def _callback(x, *args, **kwargs):
        result = f(x)
#         print(result)
        print(result, end='\r')
        logged_results.append(result)
    return _callback

def stochastic_jump(x, scale_std=0.001):
    norm_x = np.linalg.norm(x)
    delta_x = np.random.normal(0.0, norm_x * scale_std, x.shape)
    return x + delta_x

def plot_gd(h_f):
    plt.figure()
    plt.xlabel('Gradient Descent Step')
    plt.ylabel('Inequality Target')
    plt.title('Violations')
    plt.plot(np.arange(len(h_f)), h_f)
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.axis([0, len(h_f), h_f.min(), h_f.max()])
    plt.grid(True)
    plt.show()

def plot_results(results):
    results = np.asarray(results)
    plt.figure()
    plt.xlabel('Step')
    plt.ylabel('Inequality Target')
    plt.title('Minimize Results')
    plt.plot(np.arange(len(results)), results)
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.axis([0, len(results), results.min(), results.max()])
    plt.grid(True)
    plt.show()

def CHSH_Target(pds):
    pd1, pd2, pd3, pd4 = pds
    return -(pd1.correlation(('A', 'B')) + pd2.correlation(('A', 'B')) + pd3.correlation(('A', 'B')) - pd4.correlation(('A', 'B')))

def convex_dist(rvc, param):
    param = param.copy()
    param = np.abs(param)
    param /= np.sum(param)
    param = param.reshape(rvc.outcome_space.get_input_base())
    return ProbDist(rvc, param)

def convex_seed(rvc):
    return np.random.uniform(0,1,4*len(rvc.outcome_space))

class CHSHConvexityCaller(ParamCaller):

    def __init__(self):
        super().__init__(CHSH_Target, [4, 4, 4, 4])
        self.rvc = rvc

    def context(self, param):
        return tuple(convex_dist(self.rvc, param[self.slots[i]]) for i in range(4))

if __name__ == '__main__':
    for i in range(2):
        f = CHSHConvexityCaller()
        x0 = convex_seed(rvc)
        optimize.minimize(f, x0, tol=0.00001, callback=minimize_callback(f))

    print(len(result_backlog))

    result_list = [np.array(result_backlog[i]) * -1 for i in range(len(result_backlog))]

    plt.figure(figsize=(4.0, 3.0))
    plt.xlabel(r'Optimization Step')
    plt.ylabel(r'Inequality Target $\mathcal{I}_{{CHSH}}$')
    plt.title('Convex Optimizations \n Against CHSH Inequality')
    max_step = -np.inf
    min_h = int(2 * min(result_list[i][0] for i in range(len(result_list))))/2
    max_h = 4.5
    plt.axhline(4, linestyle='--', color='black')
    for i in range(len(result_list)):
        j = 0
        while j < len(result_list[i]):

            if (4 - result_list[i][j]) < 0.001:
                if j > max_step:
                    max_step = j
                break
            j += 1
    print(max_step)
    for i in range(len(result_list)):
        y = result_list[i][0:max_step]
        plt.plot(np.arange(len(y)), y)
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.axis([0, max_step-1, min_h, max_h])
    plt.grid(True)
    # plt.savefig(utils.temp_dir('example.pdf'))
    # plt.savefig(r"T:\GitProjects\hardy-fritz-triangle-paper\figures\example.pgf", format='pgf', bbox='tight')
    plt.savefig(r"T:\GitProjects\hardy-fritz-triangle-paper\figures\example.pdf", format='pdf', dpi=1000, bbox='tight')
    # tikz_save(
    #     utils.temp_dir('CHSH_convex.tex'),
    #     figureheight = '\\figureheight',
    #     figurewidth = '\\figurewidth')
    # plt.show()


