"""
Contains classes and utilities associated with random variables.
"""
import itertools
import numpy as np
import re
from ..utilities.timing_profiler import timing
from ..utilities.sort_lookup import SortLookup
from ..utilities import utils
from . import variable_sort

_rv_sep = '_'
_rv_no_sep_match = r"([a-z]+)([0-9]+)"

def split_name(symbolic):
    if _rv_sep in symbolic:
        base_name, name_modifier = symbolic.split(_rv_sep, 1)
    else:
        match = re.match(_rv_no_sep_match, symbolic, re.I)
        if match:
            base_name, name_modifier = match.groups()
        else:
            base_name, name_modifier = symbolic, None
    return base_name, name_modifier

class RandomVariable():

    def __init__(self, name, outcome_desc):
        self.base_name, self.name_modifier = split_name(name)
        self.name = name
        if isinstance(outcome_desc, int):
            self.outcomes = list(range(outcome_desc))
        else:
            self.outcomes = outcome_desc
        self.num_outcomes = len(self.outcomes)

    # def outcome_label(self, index):
    #     return self.outcomes[index]

    # def label_index(self, label):
    #     return self.outcomes.index(label)

    def __repr__(self):
        _repr = "{name}: {0} -> {1}".format(self.num_outcomes, self.outcomes, name=self.name)
        return _repr

    def __str__(self):
        return self.__repr__()

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
        self._name_lookup = dict((rv.name, rv) for rv in super().__iter__())
        self._names = SortLookup(variable_sort.sort(self._name_lookup.keys()))

    def outcome_space(self):
        outcome_space = itertools.product(
            *[self._name_lookup[rv_name].outcomes for rv_name in self._names.list]
        )
        outcome_index_space = itertools.product(
            *[range(self._name_lookup[rv_name].num_outcomes) for rv_name in self._names.list]
        )
        for outcome_index, outcome in zip(outcome_index_space, outcome_space):
            yield self._names.list, outcome_index, outcome

    def get_rvs(self, names):
        """ Get random variables from a list of names """
        if not isinstance(names, (list, tuple)):
            names = [names]
        return [self.get_rv(name) for name in names]

    def get_rv(self, name):
        """ Get Random Variable from single name """
        if name in self._name_lookup:
            return self._name_lookup[name]
        return None

    def sub(self, names):
        return RandomVariableCollection(self.get_rvs(names))

    def __iter__(self):
        for i in self._names.list:
            yield self.get_rv(i)

    def __str__(self):
        print_list = []
        print_list.append("RandomVariableCollection")
        print_list.append('{0} Random Variables:'.format(len(self)))
        print_list.append('Outcomes: {0}'.format(utils.factorization([rv.num_outcomes for rv in self])))
        for rv in self:
            print_list.append(rv.__repr__())
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