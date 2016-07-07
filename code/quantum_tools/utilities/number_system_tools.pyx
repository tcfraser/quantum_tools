cimport cython
from libc.stdlib cimport malloc, free

ctypedef long nstype

cdef nstype* tuple_process(tup, nstype size):

    cdef nstype *typed_tuple
    cdef nstype i = 0

    typed_tuple = <nstype *>malloc(size*cython.sizeof(nstype))
    if typed_tuple is NULL:
        raise MemoryError()

    for i in range(size):
        typed_tuple[i] = tup[i]

    #convert back to python return type
    return typed_tuple

def shift_sum(nstype shift, digits, base):
    cdef nstype size = len(digits)
    cdef nstype* c_digits = tuple_process(digits, size)
    cdef nstype* c_base = tuple_process(base, size)
    cdef nstype val = c_shift_sum(shift, c_digits, c_base, size)
    with nogil:
        free(c_digits)
        free(c_base)
    return val

cdef nstype c_shift_sum(nstype shift, nstype* digits, nstype* base, nstype size):
    cdef nstype val = shift
    cdef nstype i = 0
    for i in range(size):
        val += digits[i] * base[i]
    return val