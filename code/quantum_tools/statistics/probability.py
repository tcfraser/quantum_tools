"""
Contains the abstract class of a probability distrobution and associated marginals.
"""
import numpy as np
from ..utilities import utils
from ..utilities.profiler import profile
from .variable import RandomVariableCollection

class ProbDist():

    def __init__(self, rvc, support, axis_map=None):
        self._support = np.asarray(support)
        if (self._support.size == 0):
            self.__is_null = True
        else:
            self.__is_null = False
        # print(self._support > 0)
        assert(np.all(self._support >= 0))
        if not self.__is_null:
            # print(rvc)
            # print(support)
            # print(np.sum(self._support))
            sum_support = np.sum(self._support)
            assert(utils.is_close(sum_support,1.0)), "Probability distribution does not sum to 1.0. Sums to {0}".format(sum_support)
        self._rvc = rvc
        self._num_variables = len(self._rvc)
        if axis_map is not None:
            self._axis_map = axis_map
        else:
            # Assume order matches canonical order
            self._axis_map = self._rvc.names.dict

    def _multi_axis_map(self, names):
        return tuple(self._axis_map[name] for name in names)

    @staticmethod
    def from_callable_support(rvc, callable_support):
        array_dims = tuple(rv.num_outcomes for rv in rvc)
        # print(array_dims)
        support = np.zeros(array_dims)
        for _, outcome_index, outcome in rvc.outcome_space():
            p = callable_support(*outcome)
            support[outcome_index] = p
        return ProbDist(rvc, support, axis_map=None)

    def __mul__(A, B):
        if A.__is_null:
            return B
        if B.__is_null:
            return A
        Av = A._rvc
        Bv = B._rvc
        As = A._support
        Bs = B._support
        if Av.intersection(Bv):
            raise NotImplemented("Can't multiply distributions that share random variables.")
        ABv = Av.union(Bv)
        ABs = np.zeros(As.shape + Bs.shape)
        ABs = np.tensordot(As, Bs, axes=0)
        # print(As.shape + Bs.shape)
        # print(ABs.shape)
        assert(ABs.shape == As.shape + Bs.shape)
        AB_axis_map = {}
        AB_axis_map.update(A._axis_map)
        for rv_name, axis in B._axis_map.items():
            AB_axis_map[rv_name] = axis+len(As.shape)
        # print(A._axis_map)
        # print(B._axis_map)
        # print(AB_axis_map)
        AB = ProbDist(ABv, ABs, axis_map=AB_axis_map)
        return AB

    def _sub_support(self, outcome_dict):
        outcome_association = []
        slice_obj = [slice(None)] * self._num_variables
        for rv_name, outcome_label in outcome_dict.items():

            rv = self._rvc.get_rv(rv_name)
            outcome_index = rv.outcomes.index(outcome_label)
            slice_obj[self._axis_map[rv_name]] = outcome_index
            outcome_association.append((rv, outcome_index))
        sub_support = self._support[slice_obj]
        return sub_support, outcome_association

    def prob(self, outcome_dict):
        sub_support, _ = self._sub_support(outcome_dict)
        prob = np.sum(sub_support)
        return prob

    def _total_prob(self, selectors):
        total = 0.0
        for i in selectors:
            total += self._support[i]
        return total

    def coincidence(self, rvc_names, method='same'):
        rvc = self._rvc.sub(rvc_names)
        return self._coincidence(rvc, method)

    def _coincidence(self, rvc, method):
        mpd = self._marginal(rvc)
        additive = []
        subtractive = []
        for names, outcome_indices, outcome in rvc.outcome_space():
            slice_indices = tuple(outcome_indices[i] for i in mpd._multi_axis_map(names))
            if method=='same':
                if utils.all_equal(outcome):
                    additive.append(slice_indices)
                else:
                    subtractive.append(slice_indices)
            if method=='two_expect':
                # assume outcomes mean {-1, +1}
                product_outcome = (-1)**(len(outcome) - sum(outcome))
                if product_outcome == 1:
                    additive.append(slice_indices)
                else:
                    subtractive.append(slice_indices)
        # print(additive)
        # print(subtractive)
        add_prob = mpd._total_prob(additive)
        sub_prob = mpd._total_prob(subtractive)
        return add_prob - sub_prob

    def condition(self, cond_outcome_dict):
        sub_support, outcome_association = self._sub_support(cond_outcome_dict)
        prob = np.sum(sub_support)
        norm_sub_support = sub_support / prob
        rvs = [rv_assoc[0] for rv_assoc in outcome_association]
        d_rvc = self._rvc - RandomVariableCollection(rvs)
        conditioned_pd = ProbDist(d_rvc, norm_sub_support)
        return conditioned_pd

    def marginal(self, rvc_names):
        rvc = self._rvc.sub(rvc_names)
        return self._marginal(rvc)

    def _marginal(self, rvc):
        rv_to_sum = self._rvc - rvc
        if not rv_to_sum:
            return self
        else:
            axis_to_keep = [self._axis_map[rv.name] for rv in rvc]
            axis_to_sum = tuple(set(range(self._support.ndim)) - set(axis_to_keep))
            marginal_support = np.sum(self._support, axis=axis_to_sum)
            marginal_pd = ProbDist(rvc, marginal_support)
            return marginal_pd

    def H(self, *args):
        return self.entropy(*args)

    def entropy(self, X_names, Y_names=()):
        X = self._rvc.sub(X_names)
        Y = self._rvc.sub(Y_names)
        return self._entropy(X, Y)

    def _entropy(self, X, Y):
        XY = X.union(Y)
        p_XY = self._marginal(XY)
        p_Y = self._marginal(Y)
        H_XY = utils.entropy(p_XY._support)
        H_Y = utils.entropy(p_Y._support)
        # Chain rule H(X|Y) = H(Y, X) - H(Y)
        return H_XY - H_Y

    def I(self, *args):
        return self.mutual_information(*args)

    def mutual_information(self, Xs_names, Y_names=[]):
        Xs = tuple(self._rvc.sub(X_names) for X_names in Xs_names)
        Y = self._rvc.sub(Y_names)
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

    def _unpack_prob_space(self):
        for names, indices, outcome in self._rvc.outcome_space():
            axis_desc = self._multi_axis_map(names)
            support_slice = [0]*len(names)
            for i, slice_index in enumerate(axis_desc):
                support_slice[slice_index] = indices[i]
            p = self._support[tuple(support_slice)]
            yield outcome, p

    def canonical_ravel(self):
        for outcome, p in self._unpack_prob_space():
            yield p

    def __str__(self):
        fs = "{outcome} -> {probability}"
        print_list = ["=== ProbDist ==="]
        # print_list.append(self.__repr__())
        print_list.append(str(self._rvc))
        print_list.append(fs)
        for outcome, p in self._unpack_prob_space():
            print_list.append(fs.format(outcome=outcome, probability=p))
        print_list.append("================")
        return '\n'.join(print_list)

@profile
def perform_tests():
    rvc = RandomVariableCollection({})
    support = []
    pd = ProbDist(rvc, support)
    print(pd)
    print(pd.prob({}))

if __name__ == '__main__':
    perform_tests()