"""
Contains classes and utilities associated with random variables.
"""
import itertools
import numpy as np
from ..utilities.timing_profiler import timing
from ..utilities.sorted_nicely import sorted_nicely

class RandomVariable():

    def __init__(self, name, outcome_desc):
        self.name = name
        if isinstance(outcome_desc, int):
            self.outcomes = list(range(outcome_desc))
        else:
            self.outcomes = outcome_desc
        self.num_outcomes = len(self.outcomes)

    def outcome_label(self, index):
        return self.outcomes[index]

    def label_index(self, label):
        return self.outcomes.index(label)

    def __str__(self):
        _repr = "{name}: {0} outcomes -> {1}".format(self.num_outcomes, self.outcomes, name=self.name)
        print_list = []
        print_list.append(self.__repr__())
        print_list.append(_repr)
        return '\n'.join(print_list)

def rvc(names, outcome_descs):
    rvs = []
    for i, name in enumerate(names):
        rvs.append(RandomVariable(name, outcome_descs[i]))
    return RandomVariableCollection(rvs)

class RandomVariableCollection(set):

    def __init__(self, rvs=()):
        if isinstance(rvs, RandomVariable):
            rvs = [rvs]
        if rvs is None:
            rvs = []
        rvs = list(filter(None, rvs))
        super(RandomVariableCollection, self).__init__(rvs)
        self._index_map = sorted_nicely([rv.name for rv in rvs])
        # print(self._index_map)
        self.outcome_indices = list(itertools.product(*(np.arange(rv.num_outcomes) for rv in self)))
        self.outcomes = list(itertools.product(*(rv.outcomes for rv in self)))
        self.outcome_zip = list(zip(self.outcome_indices, self.outcomes))

    def indices(self, rvc):
        lst = []
        for rv in rvc:
            if rv.name in self._index_map:
                lst.append(self.index(rv))
        return lst

    def index(self, rv):
        return self._index_map.index(rv.name)

    def get(self, *args):
        if len(args) == 1:
            return next((rv for rv in super().__iter__() if rv.name == args[0]), None)
        else:
            return [self.get(arg) for arg in args]

    def sub(self, *args):
        return RandomVariableCollection(self.get(*args))

    def __iter__(self):
        for i in self._index_map:
            yield self.get(i)

    def __getitem__(self, slice):
        return self.get(*self._index_map[slice])

    def __str__(self):
        print_list = []
        print_list.append("RandomVariableCollection")
        print_list.append('{0} Random Variables:'.format(len(self)))
        rv_names = [rv.name for rv in self]
        rv_outcomes = [str(rv.num_outcomes) for rv in self]
        if len(rv_names) > 0:
            print_list.append(', '.join(rv_names))
        if len(rv_outcomes) > 0:
            print_list.append(', '.join(rv_outcomes))
        return '\n'.join(print_list)

    @classmethod
    def _wrap_methods(cls, names):
        def wrap_method_closure(name):
            def inner(self, *args):
                result = getattr(super(cls, self), name)(*args)
                if isinstance(result, set):
                    result = cls(result)
                return result
            inner.fn_name = name
            setattr(cls, name, inner)
        for name in names:
            wrap_method_closure(name)

RandomVariableCollection._wrap_methods(['__ror__', 'difference_update', '__isub__',
    'symmetric_difference', '__rsub__', '__and__', '__rand__', 'intersection',
    'difference', '__iand__', 'union', '__ixor__',
    'symmetric_difference_update', '__or__', 'copy', '__rxor__',
    'intersection_update', '__xor__', '__ior__', '__sub__',
])

@timing
def perform_tests():
    A = RandomVariable('A', 2)
    A1 = RandomVariable('A1', ['x', 'y', 'z'])
    A2 = RandomVariable('A2', 2)
    B = RandomVariable('B', 2)
    B1 = RandomVariable('B1', 2)
    B2 = RandomVariable('B2', 2)
    print(A)
    print(A1)

    AB = RandomVariableCollection([A, B, A2, A1])
    print(AB)
    # As = RandomVariableCollection([A, A1, A2])
    # Bs = RandomVariableCollection([B, B1, B2])
    # print(list(AB.outcome_indices))
    # print(list(AB.outcomes))
    # for i in AB:
    #     print(i)
    # print(AB)
    # print((AB - As))
    # if not AB - AB:
    #     print("AB - AB is nothing")
    # print(AB('A'))
    # print(AB('A_1'))
    # print(AB(('A', 1)))
    # print(AB(('A', 1), ('A', 2)))
    # print(AB('A_1', 'A_2'))


if __name__ == '__main__':
    perform_tests()