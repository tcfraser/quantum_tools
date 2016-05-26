"""
Contains the abstract class of a probability distrobution and associated marginals.
"""
import inspect
import numpy as np
import itertools
import copy
from utils import Utils

class ProbDistro():

    def __init__(self):
        self._is_initialized = False
        self._callable_support = None
        self._cached_support = None
        self._has_cached_support = False

    def __init_final__(self, num_variables):
        self._is_initialized = True
        self._num_variables = num_variables
        self._variable_space = np.arange(num_variables)
        self._indiv_outcome_space = np.arange(self._num_outcomes)
        self._variable_set = set(self._variable_space)
        self._outcome_space = itertools.product(*[self._indiv_outcome_space]*self._num_variables)

    @staticmethod
    def from_callable_support(support, num_outcomes, num_variables=-1):
        pd = ProbDistro()
        pd._num_outcomes = num_outcomes
        pd._callable_support = support
        if num_variables < 0:
            num_variables = len(inspect.getargspec(pd._callable_support).args)
        else:
            num_variables = num_variables
        pd.__init_final__(num_variables)
        return pd

    @staticmethod
    def from_cached_support(support):
        pd = ProbDistro()
        support = np.asarray(support)
        support_shape = support.shape
        if support_shape:
            support_max_shape = max(support_shape)
            pad = tuple((0, support_max_shape-orig_shape) for orig_shape in support_shape)
            pd._cached_support = np.pad(support, pad_width=pad, mode='constant', constant_values=0)
            pd._num_outcomes = support_max_shape
        else:
            pd._cached_support = support
            pd._num_outcomes = 0
        num_variables = support.ndim
        pd._has_cached_support = True
        pd.__init_final__(num_variables)
        return pd

    def __call__(self, *args):
        assert(len(args) == self._num_variables), "Number of variables doesn't match that of support."
        if self._has_cached_support:
            return self._cached_support[args]
        else:
            return self._callable_support(*args)

    def cache_support(self, in_place=False):
        if self._has_cached_support:
            if in_place:
                return self
            else:
                return copy.deepcopy(self)
        array_dims = tuple(self._num_outcomes for i in range(self._num_variables))
        support = np.zeros(array_dims)
        for index, _ in np.ndenumerate(support):
            support[index] = self._callable_support(*index)
        # print(support)
        if in_place:
            self._cached_support = support
            self._has_cached_support = True
            return self
        else:
            cached_support = ProbDistro.from_cached_support(support)
            return cached_support

    def condition(self, on, vals):
        on = Utils.en_tuple(on)
        on_set = set(on)
        assert(on_set.issubset(self._variable_set))
        # diff_set = self._variable_set.difference(on_set)
        self.cache_support(in_place=True)
        index = [slice(None)] * self._num_variables
        for on_index, on_variable in enumerate(on):
            index[on_variable] = vals[on_index]
        sub_cached_support = self._cached_support[index]
        prob = np.sum(sub_cached_support)
        norm_sub_cached_support = sub_cached_support / prob

        cpd = ProbDistro.from_cached_support(norm_sub_cached_support)
        return cpd

    def entropy(self, *args):
        return entropy(self, *args)

    def mutual_information(self, *args):
        return mutual_information(self, *args)

    def marginal(self, on):
        on_set = Utils.en_set(on)
        assert(on_set.issubset(self._variable_set))
        num_removal = len(on_set)
        num_marginal_args = self._num_variables - num_removal
        cached_pd = self.cache_support()
        marginal_cached_support = np.sum(cached_pd._cached_support, axis=tuple(on_set))
        mpd = ProbDistro.from_cached_support(marginal_cached_support)
        return mpd
        # marginal_args = args
        # def marginal_support(*args):
        #     marginal_sum = 0.0
        #            marginal_sum
        #     for outcome in itertools.product(*[range(self._num_outcomes)]*num_removal):
        #         outcome_args = list[args]
        #         for i in range(self._num_variables):
        #             if i in marginal_args:
        #                 outcome_args.
        #         for
        #     return
        # mpd = ProbDistro.from_callable_support(marginal_support, num_outcomes=self._num_outcomes, num_variables=num_marginal_args)
        # if self._has_cached_support:
        #     mpd = mpd.cache_support()

    def __str__(self):
        fs = "{outcome} → {probability}"
        print_list = []
        print_list.append(self.__repr__())
        print_list.append("Outcomes Per Variable: {0}".format(self._num_outcomes))
        print_list.append("Number of Variables: {0}".format(self._num_variables))
        if self._has_cached_support:
            print_list.append("Cached Support")
        else:
            print_list.append("Callable Support")
        print_list.append(fs)
        for outcome in self._outcome_space:
            print_list.append(fs.format(probability=self(*outcome), outcome=outcome))
        return '\n'.join(print_list)

def pd_c(pd, u):
    u_set = Utils.en_set(u)
    return pd._variable_set.difference(u_set)

def entropy(pd, u=None):
    u = Utils.en_set(u)
    mpd = pd.marginal(pd_c(pd, u))
    mpd.cache_support(in_place=True)
    entropy_array = Utils.v_entropy(mpd._cached_support)
    entropy_val = np.sum(entropy_array)
    return entropy_val

def mutual_information(pd, u, v):
    u = Utils.en_set(u)
    v = Utils.en_set(v)
    uv = u.union(v)
    assert(not u.intersection(v))
    # assert(not cu.intersection(cv))
    uv_entropy = entropy(pd, uv)
    u_entropy = entropy(pd, u)
    v_entropy = entropy(pd, v)
    return u_entropy + v_entropy - uv_entropy

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
    #   1  |  0  |  1  ||  0
    #   1  |  1  |  1  ||  1/12
    def demo_distro(a,b,c):
        if (b == c == 1):
            return 1/12
        elif (a == b == c == 0):
            return 1/3
        elif (a == 1 and b == 0 and c == 0):
            return 1/2
        else:
            return 0
    pd1 = ProbDistro.from_callable_support(demo_distro, num_outcomes=2)
    pd2 = ProbDistro.from_cached_support([[0, 1/2],[1/2, 0]])
    pd3 = pd1.cache_support()
    pd4 = pd1.marginal(on=(2,))
    pd5 = pd1.marginal(on=(0,1))
    pd6 = pd1.condition(on=(0,), vals=(0,))
    # pd1(0,0)
    # print(pd1)
    # print(pd6)
    # print(pd6.entropy())
    print(pd2)
    print(pd2.marginal(on=0))
    print(pd2.marginal(on=(0,1)))
    print(pd2.marginal(on=(0,1)).entropy())
    print(entropy(pd2.marginal(on=(0,1))))
    print(pd1.marginal(on=(0)).mutual_information(0,1))
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
    # print(pd1(0,0,0))
    # print(pd3(0,0,0))
    # print(pd2(0,1))
    # print(pd2(1,1))
    # print(pd2)
    # print(pd4)
    # print(pd5)

if __name__ == '__main__':
    perform_tests()
