import numpy as np
import itertools

def get_int_base(integer_system):
    bases = integer_system[-1] + 1
    num_digits = len(bases)
    base_desc = [bases[i+1:] for i in range(num_digits)]
    bases_prods = np.array([np.prod(k) for k in base_desc])
    bases_prods_T = bases_prods.T
    # def f(integers):
    #     return np.dot(bases_prods_T, integers)
    return bases_prods_T

def perform_tests():
    a = np.array(list(itertools.product([0,1,2,3,4], [0,1,2], [0,1,2,3])))
    # print(len(a))
    int_map = build_integer_map(a)

    i = int_map(np.array([0,2,3]))
    print(i)
    print(a[i])

if __name__ == '__main__':
    perform_tests()