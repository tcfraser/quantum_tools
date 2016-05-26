"""
Contains the abstract class of a probability distrobution and associated marginals.
"""
import inspect
import numpy as np
import itertools

class ProbDistro():

    def __init__(self):
        self._is_initialized = False
        self._callable_support = None
        self._cached_support = None
        self._has_cached_support = False

    def __init_final__(self):
        self._is_initialized = True
        self._outcome_space = itertools.product(*[range(self._num_outcomes)]*self._num_variables)

    @staticmethod
    def from_callable_support(support, num_outcomes, num_variables=-1):
        pd = ProbDistro()
        pd._num_outcomes = num_outcomes
        pd._callable_support = support
        if num_variables < 0:
            pd._num_variables = len(inspect.getargspec(pd._callable_support).args)
        else:
            pd._num_variables = num_variables
        pd.__init_final__()
        return pd

    @staticmethod
    def from_cached_support(support):
        pd = ProbDistro()
        support = np.asarray(support)
        support_shape = support.shape
        support_max_shape = max(support_shape)
        pad = tuple((0, support_max_shape-orig_shape) for orig_shape in support_shape)
        pd._cached_support = np.pad(support, pad_width=pad, mode='constant', constant_values=0)
        pd._num_outcomes = support_max_shape
        pd._num_variables = support.ndim
        pd._has_cached_support = True
        pd.__init_final__()
        return pd

    def __call__(self, *args):
        assert(len(args) == self._num_variables), "Number of variables doesn't match that of support."
        if self._has_cached_support:
            return self._cached_support[args]
        else:
            return self._callable_support(*args)

    def cache_support(self):
        if self._has_cached_support:
            return self
        array_dims = tuple(self._num_outcomes for i in range(self._num_variables))
        support = np.zeros(array_dims)
        for index, _ in np.ndenumerate(support):
            support[index] = self._callable_support(*index)
        # print(support)
        cached_support = ProbDistro.from_cached_support(support)
        return cached_support

    def marginal(self, *args):
        num_removal = len(args)
        num_marginal_args = self._num_variables - num_removal
        cached_pd = self.cache_support()
        marginal_cached_support = np.sum(cached_pd._cached_support, axis=args)
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
        self.__repr__()
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
    pd4 = pd1.marginal(2)
    pd5 = pd1.marginal(0,1)
    # pd1(0,0)
    print(pd1)
    # print(pd1(0,0,0))
    # print(pd3(0,0,0))
    # print(pd2(0,1))
    # print(pd2(1,1))
    # print(pd2)
    print(pd4)
    print(pd5)

if __name__ == '__main__':
    perform_tests()
