import numpy as np
from ..inflation import marginal_equality
from ..contexts.measurement import Measurement
from ..utilities import utils
from ..utilities.profiler import profile
from ..statistics.variable import RandomVariableCollection
# from ..inflation import positive_linear_solve
from ..examples import prob_dists
from ..visualization.sparse_vis import plot_coo_matrix

sc444444 = [
    [['A1', 'B1', 'C1'], ['A4', 'B4', 'C4']],
    [['A1', 'B2', 'C3'], ['A4', 'B3', 'C2']],
    [['A2', 'B3', 'C1'], ['A3', 'B2', 'C4']],
    [['A2', 'B4', 'C3'], ['A3', 'B1', 'C2']],
    [['A1'], ['B3'], ['C4']],
    [['A1'], ['B4'], ['C2']],
    [['A2'], ['B1'], ['C4']],
    [['A2'], ['B2'], ['C2']],
    [['A3'], ['B3'], ['C3']],
    [['A3'], ['B4'], ['C1']],
    [['A4'], ['B1'], ['C3']],
    [['A4'], ['B2'], ['C1']],
]
sc444444_outcomes = [4]*12

sc222222 = [
    [['A2'], ['B2'], ['C2']],
    [['B2'], ['A2',   'C1']],
    [['C2'], ['A1',   'B2']],
    [['A2'], ['B1',   'C2']],
    [['A1',   'B1',   'C1']],
]
sc222222_outcomes = [2]*6

@profile
def go():
    symbolic_contexts = sc444444
    outcomes = sc444444_outcomes
    inflation_rvc = RandomVariableCollection.new(names=marginal_equality.rv_names_from_sc(symbolic_contexts), outcomes=outcomes)
    print(inflation_rvc)
    # inflation_rvc = RandomVariableCollection.new(names=marginal_equality.rv_names_from_sc(symbolic_contexts), outcomes=[4]*12)
    m = marginal_equality.marginal_mtrx(inflation_rvc, symbolic_contexts)
    plot_coo_matrix(m)
    print(m.shape)
    print(m.nnz)


if __name__ == '__main__':
    go()
