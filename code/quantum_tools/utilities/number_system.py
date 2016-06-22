import numpy as np
from .binary_tree import PositivityTree
from operator import mul
from collections import namedtuple
from functools import reduce
from .number_system_tools import shift_sum

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

class MultiBaseLookupBTOptimized():

    def __init__(self):
        self.__bases = PositivityTree()
        self.__fpbase = None
        self.__current_shift = 0

    def get_val(self, digits, use_fpb=False):
        if use_fpb:
            if self.__fpbase:
                shifted_base = self.__fpbase
            else:
                raise Exception("There is no fully positive base.")
        else:
            shifted_base = self.__bases.find(digits)
        if shifted_base is None:
            raise Exception("Digits {0} have no registered base.".format(digits))
        else:
            shift, base = shifted_base
        val = shift_sum(shift, digits, base)
        # val = shift + np.dot(digits, base) # Way longer
        return val

    def register_base(self, max_values, base_indices=None, size=0, shift=False):
        base_multipliers = tuple(reduce(mul, max_values[i+1:], 1) for i in range(len(max_values)))
        base_space_size = int(reduce(mul, max_values, 1))
        if base_indices is not None:
            assert(size > len(max_values))
            assert(len(base_indices) == len(max_values))
            filled_base_multipliers = [0]*size
            for i, bi in enumerate(base_indices):
                filled_base_multipliers[bi] = base_multipliers[i]
            base_multipliers = tuple(filled_base_multipliers)
        local_shift = 0
        if shift:
            local_shift = self.__current_shift
            self.__current_shift += base_space_size
        sb = ShiftedBase(local_shift, base_multipliers)

        path = tuple(1 if b != 0 else -1 for b in base_multipliers)
        is_fp = all(p > 0 for p in path)
        if is_fp:
            self.__fpbase = sb
        self.__bases.add_val(path, sb)
        print(sb)

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
