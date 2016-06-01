"""
Contains the abstract class of a probability distrobution and associated marginals.
"""
import inspect
import numpy as np
import itertools
import copy
from utils import Utils
from timing_profiler import timing
from variable import rvc, RandomVariableCollection

class ProbDist():

    def __init__(self, rvc, support):
        self._support = np.asarray(support)
        self._rvc = rvc
        self._num_variables = len(self._rvc)

    @staticmethod
    def from_callable_support(rvc, callable_support):
        array_dims = tuple(rv.num_outcomes for rv in rvc)
        # print(array_dims)
        support = np.zeros(array_dims)
        for index, _ in np.ndenumerate(support):
            support[index] = callable_support(*index)
        return ProbDist(rvc, support)

    def __getitem__(self, slice):
        return self._support[slice]

    def _sub_support(self, value_dict):
        value_array = []
        for key, value in value_dict.items():
            rv = self._rvc.get(key)
            value_array.append((rv, value))
        slice_obj = [slice(None)] * self._num_variables
        for rv, value in value_array:
            if value not in rv.outcome_space:
                raise Exception("Random Variable {rv.name} does not have outcome {outcome} in outcome space {rv.outcome_space}.".format(rv=rv, outcome=value))
            slice_obj[self._rvc.index(rv)] = value

        sub_support = self._support[slice_obj]
        return sub_support, value_array

    def prob(self, value_dict):
        sub_support, _ = self._sub_support(value_dict)
        prob = np.sum(sub_support)
        return prob

    def condition(self, cond_dict):
        sub_support, value_array = self._sub_support(cond_dict)
        prob = np.sum(sub_support)
        norm_sub_support = sub_support / prob
        rvs = [tup[0] for tup in value_array]
        d_rvc = self._rvc - RandomVariableCollection(rvs)
        conditioned_pd = ProbDist(d_rvc, norm_sub_support)
        return conditioned_pd

    def marginal(self, rvc_names):
        rvc = self._rvc.sub(*rvc_names)
        return self._marginal(rvc)

    def _marginal(self, rvc):
        rv_to_sum = self._rvc - rvc
        if not rv_to_sum:
            return self
        else:
            axis_to_keep = self._rvc.indices(rvc)
            axis_to_sum = tuple(set(range(self._support.ndim)) - set(axis_to_keep))
            marginal_support = np.sum(self._support, axis=axis_to_sum)
            marginal_pd = ProbDist(rvc, marginal_support)
            return marginal_pd

    def H(self, *args):
        return self.entropy(*args)

    def entropy(self, X_names, Y_names=()):
        X = self._rvc.sub(*X_names)
        Y = self._rvc.sub(*Y_names)
        return self._entropy(X, Y)

    def _entropy(self, X, Y):
        XY = X.union(Y)
        p_XY = self._marginal(XY)
        p_Y = self._marginal(Y)
        H_XY = Utils.entropy(p_XY._support)
        H_Y = Utils.entropy(p_Y._support)
        # Chain rule H(X|Y) = H(Y, X) - H(Y)
        return H_XY - H_Y

    def I(self, *args):
        return self.mutual_information(*args)

    def mutual_information(self, Xs_names, Y_names=[]):
        Xs = tuple(self._rvc.sub(*X_names) for X_names in Xs_names)
        Y = self._rvc.sub(*Y_names)
        return self._mutual_information(Xs, Y)

    def _mutual_information(self, Xs, Y):
        if len(Xs) > 1:
            Xs_reduced = Xs[0:-1]
            Y_expanded = Y.union(Xs[-1])

            return self._mutual_information(Xs_reduced, Y) - \
                   self._mutual_information(Xs_reduced, Y_expanded)
        elif len(Xs) == 1:
            return self._entropy(Xs[0], Y)
        else:
            return 0

    def ravel_support(self):
        return np.ravel(self._support)

    def __str__(self):
        fs = "{outcome} -> {probability}"
        print_list = []
        print_list.append(self.__repr__())
        print_list.append(str(self._rvc))
        for outcome in self._rvc.outcome_space:
            print_list.append(fs.format(probability=self[outcome], outcome=outcome))
        return '\n'.join(print_list)

@timing
def perform_tests():
    # if a,b,c are bits, there are 2**3 = 8 outcomes
    #   a  |  b  |  c  ||  P
    #   --------------------
    #   0  |  0  |  0  ||  1/3
    #   0  |  1  |  0  ||  0
    #   1  |  0  |  0  ||  0
    #   1  |  1  |  0  ||  1/2
    #   0  |  0  |  1  ||  0
    #   0  |  1  |  1  ||  1/12
    #   1  |  0  |  1  ||  1/12
    #   1  |  1  |  1  ||  0
    def demo_distro(a,b,c):
        if (a == c == 1):
            return 1/12
        elif (a == b == c == 0):
            return 1/3
        elif (a == 1 and b == 0 and c == 0):
            return 1/2
        else:
            return 0
    pd = ProbDist.from_callable_support(rvc(['A', 'B', 'C'], [2, 2, 2]), demo_distro)
    print(pd)
    print(pd.condition({'A': 1}))
    print(pd.prob({'A': 1, 'B': 0}))
    print(pd.prob({}))
    # print(pd.marginal(['A']))
    # print(pd.entropy(['A']))
    # print(pd.entropy('A'))
    # print(pd.mutual_information([['B'], ['A']], ['C']))
    # print(pd.entropy(['A'], ['C']) + pd.entropy(['B'], ['C']) - pd.entropy(['A', 'B'], ['C']))
    # print(pd.mutual_information([['B'], ['A']]))
    # print(pd.entropy(['A']) + pd.entropy(['B']) - pd.entropy(['A', 'B']))
    # print(pd.entropy(['B', 'C'], ['A']))
    # pd2 = ProbDist.from_cached_support([[0, 1/2],[1/2, 0]])
    # pd3 = pd.cache_support()
    # pd4 = pd.marginal(on=(2,))
    # pd5 = pd.marginal(on=(0,1))
    # pd6 = pd.condition(on=(0,), vals=(0,))
    # pd(0,0)
    # print(pd)
    # print(pd6)
    # print(pd6.entropy())
    # print(pd2)
    # print(pd2.marginal(on=0))
    # print(pd2.marginal(on=(0,1)))
    # # print(pd2.marginal(on=(0,1)).entropy())
    # # print(entropy(pd2.marginal(on=(0,1))))
    # # print(pd.marginal(on=(0)).mutual_information(0,1))
    # print(pd.mutual_information(1,2))
    # print(pd.mutual_information(1,1))
    # print(pd.marginal((0,2)).entropy())
    # # # print(pd.marginal((0,2)).entropy())
    # print(pd.marginal((0,1)).entropy())
    # print(pd.mutual_information(0,1))
    # print(pd.mutual_information(0,2))
    # print(pd.mutual_information(0,3))
    # print(pd2, on=(0,1))
    # print(pd2.marginal(on=None))
    # print(pd2)
    # print(pd2.marginal(on=1))
    # print(pd2, on=1)
    # print(mutual_information(pd2, 0,(0,1)))
    # print(entropy(pd2))
    # print(entropy(pd2.marginal(on=0)))
    # print(entropy(pd2.marginal(on=1)))
    # print(mutual_information(pd2, 0, 1))
    # print(pd2.condition(on=(0,), vals=(0,)))
    # print(pd2.condition(on=(0,), vals=(0,)).entropy())
    # print(pd2.entropy())
    # print(pd2.marginal(on=(1,)).entropy())
    # print(pd2.entropy() - pd2.marginal(on=(1,)).entropy())
    # print()
    # print(pd(0,0,0))
    # print(pd3(0,0,0))
    # print(pd2(0,1))
    # print(pd2(1,1))
    # print(pd2)
    # print(pd4)
    # print(pd5)

if __name__ == '__main__':
    perform_tests()
