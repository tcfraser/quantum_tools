from itertools import *
import numpy as np
import collections
from scipy import linalg
from scipy import sparse
from scipy import io
from functools import reduce
from operator import mul
from .constants import *
import random as rndm
import math
from ..config import *
import os
from collections import defaultdict

def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return tally

def u2pi():
    return rndm.uniform(0.0,1.0) * 2 * np.pi

def upi():
    return rndm.uniform(0.0,1.0) * np.pi

def recurse(f, n, arg):
    rf = f(arg)
    for _ in range(n-1):
        rf = f(rf)
    return rf

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def tabulate(function, start=0):
    "Return function(0), function(1), ..."
    return map(function, count(start))

def consume(iterator, n):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)

def nth(iterable, n, default=None):
    "Returns the nth item or a default value"
    return next(islice(iterable, n, None), default)

def quantify(iterable, pred=bool):
    "Count how many times the predicate is true"
    return sum(map(pred, iterable))

def padnone(iterable):
    """Returns the sequence elements and then returns None indefinitely.

    Useful for emulating the behavior of the built-in map() function.
    """
    return chain(iterable, repeat(None))

def ncycles(iterable, n):
    "Returns the sequence elements n times"
    return chain.from_iterable(repeat(tuple(iterable), n))

def dotproduct(vec1, vec2):
    return sum(map(operator.mul, vec1, vec2))

def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

def repeatfunc(func, times=None, *args):
    """Repeat calls to func with specified arguments.

    Example:  repeatfunc(random.random)
    """
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))

def multiplicity(iterable):
    seen = []
    multi = []
    i = 0
    for element in iterable:
        if element in seen:
            multi[seen.index(element)] += 1
        else:
            seen.append(element)
            multi.append(1)
    return zip(seen, multi)

def factorization(nums, dot='*'):
    s = []
    total = 0
    for num, multi in multiplicity(nums):
        if (multi == 1):
            s.append(str(num))
            total += num
        else:
            s.append("{0}^{1}".format(num, multi))
            total += num**multi
    return str(total) + ' = ' + dot.join(s)

def flip_list(lst):
    return dict((i,j) for j, i in enumerate(lst))

def partition(pred, iterable):
    'Use a predicate to partition entries into false entries and true entries'
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

def unique_justseen(iterable, key=None):
    "List unique elements, preserving order. Remember only the element just seen."
    # unique_justseen('AAAABBBCCDAABBB') --> A B C D A B
    # unique_justseen('ABBCcAD', str.lower) --> A B C A D
    return map(next, map(itemgetter(1), groupby(iterable, key)))

def iter_except(func, exception, first=None):
    """ Call a function repeatedly until an exception is raised.

    Converts a call-until-exception interface to an iterator interface.
    Like builtins.iter(func, sentinel) but uses an exception instead
    of a sentinel to end the loop.

    Examples:
        iter_except(functools.partial(heappop, h), IndexError)   # priority queue iterator
        iter_except(d.popitem, KeyError)                         # non-blocking dict iterator
        iter_except(d.popleft, IndexError)                       # non-blocking deque iterator
        iter_except(q.get_nowait, Queue.Empty)                   # loop over a producer Queue
        iter_except(s.pop, KeyError)                             # non-blocking set iterator

    """
    try:
        if first is not None:
            yield first()            # For database APIs needing an initial cast to db.first()
        while 1:
            yield func()
    except exception:
        pass

def first_true(iterable, default=False, pred=None):
    """Returns the first true value in the iterable.

    If no true value is found, returns *default*

    If *pred* is not None, returns the first item
    for which pred(item) is true.

    """
    # first_true([a,b,c], x) --> a or b or c or x
    # first_true([a,b], x, f) --> a if f(a) else b if f(b) else x
    return next(filter(pred, iterable), default)

def random_product(*args, repeat=1):
    "Random selection from itertools.product(*args, **kwds)"
    pools = [tuple(pool) for pool in args] * repeat
    return tuple(random.choice(pool) for pool in pools)

def random_permutation(iterable, r=None):
    "Random selection from itertools.permutations(iterable, r)"
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

def random_combination_with_replacement(iterable, r):
    "Random selection from itertools.combinations_with_replacement(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.randrange(n) for i in range(r))
    return tuple(pool[i] for i in indices)

def sparse_density(A):
    try:
        max_size = reduce(mul, A.shape, 1)
        density = A.nnz / max_size
        return density
    except Exception as e:
        print(str(e))
        raise Exception("Likely not a sparse matrix. A.__class__: {0}".format(A.__class__))

def bulk_file(name):
    return os.path.join(BULK_DIR, name)

def save_sparse(name, M):
    io.mmwrite(bulk_file(name), M)

def load_sparse(name):
    return io.mmread(bulk_file(name))

def ei(x):
    """ Exponential notation for complex numbers """
    return np.exp(i*x)

def tensor(*args):
    """ Implementation of tensor or kronecker product for tuple of matrices """
    return reduce(np.kron, args)

def multiply(*args):
    """ Implementation of multiple matrix multiplications between matrices """
    assert(False), "Shouldn't be using this."
    return reduce(mul, args)

def multidot(*args):
    """ Implementation of multiple dot product between matrices """
    return reduce(np.dot, args)

def is_hermitian(A):
    """ Checks if matrix A is hermitian """
    return np.allclose(A, A.conj().T)

def is_psd(A):
    """ Checks if matrix A is positive semi-definite """
    return np.all(linalg.eigvals(A) >= -mach_tol)

def is_trace_one(A):
    """ Checks if matrix A has unitary trace or not """
    return is_close(np.trace(A), 1)

def uniform_phase_components(n):
    phases = np.ones(n, dtype='complex')
    for i in range(1,n):
        phases[i] *= ei(u2pi())
    return phases

def uniform_n_sphere_metric(n):
    """
        for n = 3: returns
        (cos x, sin x cos y, sin x sin y)
        for n = 4: returns
        (cos x, sin x cos y, sin x sin y cos z, sin x sin y sin z)
    """
    angles = [u2pi() for _ in range(n-1)]
    trig_pairs = [(math.cos(angles[i]), math.sin(angles[i])) for i in range(n-1)]
    # trig_pairs = [(math.cos(u2pi()), math.sin(u2pi())) for i in range(n-1)]
    metric = np.ones(n)
    # metric = ['', '', '', '']
    for i in range(n):
        for j in range(i+1):
            if j < n-1:
                if i == j:
                    metric[i] *= trig_pairs[j][0] # cos
                    # metric[i] += 'cos({0})'.format(j) # cos
                else:
                    metric[i] *= trig_pairs[j][1] # sin
                    # metric[i] += 'sin({0})'.format(j) # sin
        # print(metric[i])
    # print(metric)
    # print(linalg.norm(metric))
    # print(trig_pairs)
    assert(is_close(linalg.norm(metric), 1.0)), "Not normalized."
    return metric

def is_close(a,b):
    """ Checks if two numbers are close with respect to a machine tolerance defined above """
    return abs(a - b) < mach_tol

def is_small(a):
    return is_close(a, 0)

def gen_memory_slots(mem_loc):
    i = 0
    slots = []
    for m_size in mem_loc:
        slots.append(np.arange(i, i+m_size))
        i += m_size
    return slots

def normalize(a):
    a /= linalg.norm(a)

def ket_to_dm(ket):
    """ Converts ket vector into density matrix rho """
    return np.outer(ket, ket.conj())

def __v_entropy(x):
    if x != 0.0:
        return -x*np.log2(x)
    else:
        return 0.0

def entropy(x):
    return np.sum(__v_entropy(x))

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

def all_equal(iterator, to=None):
    try:
        iterator = iter(iterator)
        first = next(iterator)
        if to is None:
            return all(first == rest for rest in iterator)
        else:
            return all(first == rest == to for rest in iterator)
    except StopIteration:
        return True

def all_equal_length(iterator):
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(len(first) ==len(rest) for rest in iterator)
    except StopIteration:
        return True

def list_mod(lst, mod):
    return [i % mod for i in lst]

def en_tuple(tup):
    if isinstance(tup, (tuple)):
        return tup
    elif isinstance(tup, (set, list)):
        return tuple(tup)
    else:
        return (tup,)

def en_list(lst):
    if isinstance(lst, (list)):
        return lst
    elif isinstance(lst, (set, tuple)):
        return list(lst)
    else:
        return [lst]

def en_set(lst):
    if isinstance(lst, (set)):
        return lst
    elif isinstance(lst, (list, tuple)):
        return set(lst)
    elif lst is None:
        return set()
    else:
        return set([lst])

def iterable_not_string(a):
    return not isinstance(i, str) and isinstance(i, collections.Iterable)

def get_triangle_permutation():
    perm = np.zeros((64,64), dtype=complex)
    for a in list(product(*[[0,1]]*6)):
        ket = tensor(*(qbs[a[i]] for i in (0,1,2,3,4,5)))
        bra = tensor(*(qbs[a[i]] for i in (1,2,3,4,5,0)))
        perm += np.outer(ket, bra)
    return perm

def largest_eig(M):
    size = M.shape[0]
    largest_eig = linalg.eigh(M, eigvals_only=True, eigvals=(size-1,size-1))[0]
    return largest_eig

def projectors(n):
    S = [np.zeros((n,n)) for _ in range(n)]
    for i in range(n):
        S[i][i,i] = 1
    return S

def cholesky(t):
    # James, Kwiat, Munro, and White Physical Review A 64 052312
    # T = np.array([
    #     [      t[0]       ,       0         ,      0        ,   0  ],
    #     [  t[4] + i*t[5]  ,      t[1]       ,      0        ,   0  ],
    #     [ t[10] + i*t[11] ,  t[6] + i*t[7]  ,     t[2]      ,   0  ],
    #     [ t[14] + i*t[15] , t[12] + i*t[13] , t[8] + i*t[9] , t[3] ],
    # ])
    # assert(len(t) == size**2), "t is not the correct length. [len(t) = {0}, size = {1}]".format(len(t), size)
    size = int(len(t)**(1/2))
    indices = range(size)
    t_c = 0 # Parameter t counter
    T = np.zeros((size,size), dtype=complex)
    for row in indices:
        for col in indices:
            if (row == col): # diagonal
                T[row, col] = t[t_c]
                t_c += 1
            elif (row > col): # lower diagonal
                T[row, col] = t[t_c] + i * t[t_c + 1]
                t_c += 2
            elif (col > row): # upper diagonal
                pass
    Td = T.conj().T
    g = np.dot(Td, T)
    # assert(is_hermitian(g)), "g not hermitian!"
    # assert(is_psd(g)), "g not positive semi-definite!"
    return g

def is_square(apositiveint):
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen: return False
        seen.add(x)
    return True

def partial_identities(seed_desc):
    size = sum(seed_desc)
    As = []
    j = 0
    for i, seed in enumerate(seed_desc):
        A = np.zeros((size, size))
        for _ in range(seed):
            A[j, j] = 1
            j += 1
        As.append(A)
    return As

def param_GL_C(t):
    assert(is_square(len(t)/2)), "Number of parameters needs to be a twice a square number, not {0}.".format(len(t))
    size = int((len(t)/2)**(1/2))
    t = np.asarray(t)
    GL_RR = np.reshape(t, (2, size, size))
    GL_C = GL_RR[0] + i * GL_RR[1]
    return GL_C

def get_orthogonal_pair(t):
    theta = t[0]
    phi = t[1]
    psi_1 = np.cos(theta) * qb0 - np.sin(theta) * ei(phi) * qb1
    psi_2 = np.sin(theta) * qb0 + np.cos(theta) * ei(phi) * qb1
    dm_1 = ket_to_dm(psi_1)
    dm_2 = ket_to_dm(psi_2)
    # print(dm_1)
    # print(dm_2)
    # print(dm_1 + dm_2)
    return dm_1, dm_2

__v_entropy = np.vectorize(__v_entropy)

def perform_tests():
    print(np.__version__)
    # return
    print(__v_entropy(np.array([[0.0, 0.5], [0.5, 0.0]])))

if __name__ == '__main__':
    perform_tests()
    # def norm_real_parameter(x):
    #     return np.cos(x)**2

def min_col(v):
    min_val = np.inf
    min_i = None
    for i in range(v.shape[0]):
        v_i = v.data[v.indptr[i]:v.indptr[i+1]]
        if len(v_i) == 0: # No entries
            val_v_i = 0
        else:
            val_v_i = v_i[0]
        if val_v_i < min_val:
            min_val = val_v_i
            min_i = i
    return min_i, min_val

def max_col(v):
    max_val = -np.inf
    max_i = None
    for i in range(v.shape[0]):
        v_i = v.data[v.indptr[i]:v.indptr[i+1]]
        if len(v_i) == 0: # No entries
            val_v_i = 0
        else:
            val_v_i = v_i[0]
        if val_v_i > max_val:
            max_val = val_v_i
            max_i = i
    return max_i, max_val