import numpy as np
import itertools
from operator import mul
from functools import reduce

class IntMap():

    def __init__(self, digits, dtype='int32'):
        self._digits = digits
        self._dtype = dtype
        self._space_size = int(reduce(mul, digits, 1))
        self.base_size = len(self._digits)
        self.__base = [reduce(mul, digits[i+1:], 1) for i in range(self.base_size)]

    def __len__(self):
        return self._space_size

    def get_digits(self, integer, base=None):
        if base is None:
            base = self.__base
        digits = []
        for b in base:
            digit = integer // b
            digits.append(digit)
            integer -= b * digit
        return digits

    def get_integer(self, digits, base=None):
        if base is None:
            base = self.__base
        return sum(a*b for a,b in zip(base, digits))
        # return np.dot(np.asarray(digits), base)

    def get_base(self):
        return self.__base

    def __iter__(self):
        return itertools.product(*[range(d) for d in self._digits])

def comp_mask(integers, n):
    c_mask = np.ones(n, dtype=bool)
    c_mask[integers] = False
    return c_mask

def perform_tests():
    a = np.array(list(itertools.product([0,1,2,3,4], [0,1,2], [0,1,2,3])))
    im = IntMap((5,3,4))
    print(im.get_digits(59))
    print(im.get_integer((4,2,3)))
    print(im.get_integer(np.array([[4,2,3],[4,2,3]])))


    a = np.arange(11)
    mask = np.array([0,3,4,7,8,9])
    c_mask = comp_mask(mask, len(a))
    print(a[c_mask])


    # print(len(a))
    # int_base = get_base(a[-1])
    # print(int_base)
    # print(np.dot((4,2,3), int_base))
    # print(int_digits(59, int_base))
    # i = int_map(np.array([0,2,3]))
    # print(i)
    # print(a[i])

if __name__ == '__main__':
    perform_tests()