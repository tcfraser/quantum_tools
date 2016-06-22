import numpy as np
from .binary_tree import PositivityTree
from operator import mul
from collections import namedtuple

class MultiBaseLookupBT():

    def __init__(self):
        self.__bases = PositivityTree()

    def get_val(self, digits):
        shift, base = self.__bases.find(digits)
        val = shift + sum(map(mul, digits, base))
        # val = shift + np.dot(digits, base) # Way longer
        return val

    def register_base(self, digits, base, shift=0):
        self.__bases.add_val(digits, (shift, base))

def BaseTuple(max_values, base_indices=None, size=0):
    base_multipliers = tuple(reduce(mul, max_values[i+1:], 1) for i in range(len(max_values)))
    if base_indices is None:
        return base_multipliers
    else:
        assert(size > len(max_values))
        assert(len(base_indices) == len(max_values))
        filled_base_multipliers = [0]*size
        for i, bi in enumerate(base_indices):
            filled_base_multipliers[bi] = max_values[i]
        return filled_base_multipliers

class MultiBaseLookupBTOptimized():

    def __init__(self):
        self.__bases = PositivityTree()
        self.__singleton_base = None

    def get_val(self, digits):
        if self.__singleton_base:
            shift, base = self.__singleton_base
        else:
            shift, base = self.__bases.find(digits)
        val = shift + sum(map(mul, digits, base))
        # val = shift + np.dot(digits, base) # Way longer
        return val

    def register_base(self, digits, base, shift=0):
        sb = ShiftedBase(shift, base)
        if self.__singleton_base == None:
            self.__singleton_base = sb
        else:
            self.__singleton_base = False
        self.__bases.add_val(digits, sb)

class MultiBaseLookupNP():

    def __init__(self, num_digits, dtype='int16'):
        self.__dtype = dtype
        self.__num_digits = num_digits
        self.__shifted_bases = [None]
        self.__base_lookup = np.zeros((2,)*num_digits, dtype=self.__dtype)

    def _get_positive_locations(self, digits):
        positive_locations = np.asarray(digits >= 0, dtype=self.__dtype)
        return positive_locations

    def get_base_for_digits(self, digits):
        return self.get_base_at(self._get_positive_locations(digits))

    def get_base_at(self, lookup_coords):
        base_posn = self.__base_lookup[tuple(lookup_coords)]
        return self.__shifted_bases[base_posn]

    def get_val(self, digits):
        shift, base = self.get_base_for_digits(digits)
        return shift + np.dot(base,digits)

    def register_base(self, digits, base, shift=0):
        sb = ShiftedBase(shift, np.asarray(base, dtype=self.__dtype))
        self.__base_lookup[tuple(self._get_positive_locations(digits))] = len(self.__shifted_bases)
        self.__shifted_bases.append(sb)
        return sb

ShiftedBase = namedtuple('ShiftedBase', ['shift', 'base'])




# cdef class IntBase():


# cdef class OffsetIntBase(IntBase):
