import numpy as np
import itertools
from operator import mul
from functools import reduce

def get_digits(integer, base):
    digits = []
    for b in base:
        digit = integer // b
        digits.append(digit)
        integer -= b * digit
    return digits

def get_integer(digits, base):
    tot = 0
    for i in range(len(base)):
        tot += digits[i] * base[i]
    return tot

class IntMap():

    # def __init__(self, input_base, dtype='int32'):
    def __init__(self, input_base):
        self.__input_base = input_base
        # self._dtype = dtype
        self.__space_size = int(reduce(mul, input_base, 1))
        self.__base_size = len(input_base)
        self.__base = [reduce(mul, input_base[i+1:], 1) for i in range(self.__base_size)]
        self.__cached_iter = None

    def __len__(self):
        return self.__space_size

    def get_digits(self, integer):
        return get_digits(integer, self.get_base())

    def get_integer(self, digits):
        return get_integer(digits, self.get_base())

    def get_base(self):
        return self.__base

    def __iter__(self):
        return itertools.product(*[range(d) for d in self.__input_base])

    def cached_iter(self):
        if self.__cached_iter is None:
            self.__cached_iter = np.array(list(iter(self)))
        return self.__cached_iter

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