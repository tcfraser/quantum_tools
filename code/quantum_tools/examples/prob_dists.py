import itertools
from ..utilities.timing_profiler import timing
from ..statistics.probability import ProbDist
from ..statistics.variable import rvc, RandomVariableCollection
from ..utilities.utils import *

def spekkens():
    perm123 = list(itertools.permutations([1,2,3]))
    rest = list(flatten([unique_everseen(itertools.permutations([0,i,i])) for i in range(4)]))
    allowed_outcomes = perm123 + rest
    support = np.zeros((4,4,4))
    p = 1/len(allowed_outcomes)
    for i in allowed_outcomes:
        support[i] = p
    # print(support)
    random_variables = rvc(['A', 'B', 'C'], [4,4,4])
    return ProbDist(random_variables, support)

def _demo_distro(a,b,c):
# if a,b,c are bits, there are 2**3 = 8 outcomes
#     a   |    b   |    c   ||  P
#   --------------------------------
#   'a'   |   'a'  |  'c1'  ||  1/3
#   'a'   |   'a'  |  'c2'  ||  1/12
#   'a'   |  'b1'  |  'c1'  ||  0
#   'a'   |  'b1'  |  'c2'  ||  1/12
#   'a2'  |   'a'  |  'c1'  ||  0
#   'a2'  |   'a'  |  'c2'  ||  0
#   'a2'  |  'b2'  |  'c1'  ||  1/2
#   'a2'  |  'b2'  |  'c2'  ||  0
    if (a == 'a' and c == 'c2' and b != 'b2'):
        return 1/12
    elif (a == 'a' and b == 'a' and c == 'c1'):
        return 1/3
    elif (a == 'a2' and b == 'b2' and c == 'c1'):
        return 1/2
    else:
        return 0

def demo_distro():
    demo_distro = ProbDist.from_callable_support(
            rvc(['A', 'B', 'C'], [['a', 'a2'], ['a', 'b1', 'b2'], ['c1', 'c2']]),
            _demo_distro
        )
    return demo_distro

def null():
    rvc = RandomVariableCollection({})
    support = []
    pd = ProbDist(rvc, support)
    return pd

@timing
def perform_tests():
    spd = spekkens()
    print(spd)
    npd = null()
    print(npd)
    print(spd * npd)
    # print(spd.H('B', 'B'))
    # print(spd.I(['B', 'B'], ['B']))
    return
    pd = demo_distro
    print(pd)
    # print(pd.condition({'A': 'a'}))
    # print(pd.prob({'A': 'a', 'B': 'b1'}))
    # print(pd.prob({'A': 'a', 'C': 'c2'}))
    print(pd.coincidence(['A', 'B']))

if __name__ == '__main__':
    perform_tests()