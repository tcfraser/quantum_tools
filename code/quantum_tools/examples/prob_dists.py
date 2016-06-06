import itertools
from ..utilities.profiler import profile
from ..statistics.probability import ProbDist
from ..statistics.variable import RandomVariableCollection
from ..utilities.utils import *
from ..utilities.constants import *

def spekkens():
    perm123 = list(itertools.permutations([1,2,3]))
    rest = list(flatten([unique_everseen(itertools.permutations([0,i,i])) for i in range(4)]))
    allowed_outcomes = perm123 + rest
    support = np.zeros((4,4,4))
    p = 1/len(allowed_outcomes)
    for i in allowed_outcomes:
        support[i] = p
    # print(support)
    random_variables = RandomVariableCollection.new(['A', 'B', 'C'], [4,4,4])
    return ProbDist(random_variables, support)

def fritz():
    pass # TODO
    # rho0 = utils.ket_to_dm(qb0)
    # rho1 = utils.ket_to_dm(qb1)
    # phi1 = utils.ket_to_dm([0.5**0.5, 0.5**0.5])
    # phi2 = utils.ket_to_dm([0.5**0.5, 0.5**0.5])

    # A = Measurement.Strats.Random.pvms_uniform(4)
    # B = Measurement.Strats.Random.pvms_uniform(4)
    # C = Measurement.Strats.Random.pvms_uniform(4)
    # rhoAB = State.Strats.Random.pure_uniform(4)
    # rhoBC = State.Strats.Random.pure_uniform(4)
    # rhoAC = State.Strats.Random.pure_uniform(4)
    # # rhoAB = State.Strats.Deterministic.mebs(2)
    # # rhoBC = State.Strats.Deterministic.mebs(2)
    # # rhoAC = State.Strats.Deterministic.mebs(2)
    # # print(A[4].shape)
    # # print(rhoBC.shape)
    # qc = QuantumContext(random_variables=rvc, measurements=(A,B,C), states=(rhoAB, rhoBC, rhoAC), permutation=perm)
    # pd = QuantumProbDist(qc)
    # return pd

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
            RandomVariableCollection.new(['A', 'B', 'C'], [['a', 'a2'], ['a', 'b1', 'b2'], ['c1', 'c2']]),
            _demo_distro
        )
    return demo_distro

def null():
    rvc = RandomVariableCollection({})
    support = []
    pd = ProbDist(rvc, support)
    return pd

@profile
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