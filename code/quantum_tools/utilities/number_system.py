import numpy as np
from .binary_tree import PositivityTree
from operator import mul

class MultiBaseLookup():

    def __init__(self):
        self.__bases = {}

    # def get_positive_locations(self, digits):
    #     positive_locations = np.asarray(digits >= 0, dtype=self.__dtype)
    #     return positive_locations

    # def get_base(self, digits):
    #     return self.get_base_at(self.get_positive_locations(digits))

    def get_val(self, digits):
        shift, base = self.__bases.find(digits)
        val = shift + sum(map(mul, digits, base))
        return val

    def register_base(self, digits, base, shift=0):
        self.__bases.add_val(digits, (shift, base))
        # self.__bases[tuple(self.get_positive_locations(digits))] = base

class MultiBaseLookup():

    def __init__(self):
        self.__bases = {}

    def get_positive_locations(self, digits):
        positive_locations = np.asarray(digits >= 0, dtype=self.__dtype)
        return positive_locations

    def get_base(self, digits):
        return self.get_base_at(self.get_positive_locations(digits))

    def get_val(self, digits):
        shift, base = self.__bases.find(digits)
        val = shift + sum(map(mul, digits, base))
        return val

    def register_base(self, digits, base, shift=0):
        self.__bases.add_val(digits, (shift, base))
        # self.__bases[tuple(self.get_positive_locations(digits))] = base

# class BinaryTree():




# cdef class IntBase():


# cdef class OffsetIntBase(IntBase):
