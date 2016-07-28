import itertools
from ..utilities.profiler import profile
from ..statistics.probability import ProbDist
from ..statistics.variable import RandomVariableCollection
from ..utilities import utils
from ..contexts.measurement import Measurement
from ..contexts.state import State
from ..contexts.quantum_context import QuantumContext, QuantumProbDist, QuantumProbDistOptimized
from ..utilities.constants import *

def perfect_correlation(rvc):
    shape = rvc.outcome_space.get_input_base()
    support = np.zeros(shape)
    assert(utils.all_equal(shape))
    for i in range(shape[0]):
        support[tuple(i for _ in range(len(shape)))] = 1
    support /= np.sum(support)
    pd = ProbDist(rvc, support)
    return pd

def uniform_qdistro(rvc, dimensions):
    if dimensions**2 == 4:
        A = Measurement.Strats.Random.pvms_uniform(4)
        B = Measurement.Strats.Random.pvms_uniform(4)
        C = Measurement.Strats.Random.pvms_uniform(4)
    else:
        A = Measurement.Strats.Random.pvms_outcomes(4, dimensions**2)
        B = Measurement.Strats.Random.pvms_outcomes(4, dimensions**2)
        C = Measurement.Strats.Random.pvms_outcomes(4, dimensions**2)
    rhoAB = State.Strats.Random.pure_uniform(dimensions**2)
    rhoBC = State.Strats.Random.pure_uniform(dimensions**2)
    rhoAC = State.Strats.Random.pure_uniform(dimensions**2)
        
    qc = QuantumContext(
        random_variables=rvc,
        measurements=(A,B,C),
        states=(rhoAB, rhoBC, rhoAC),
        permutation=utils.get_triangle_permutation(dimensions)
    )
    if dimensions**2 == 4:
        pd = QuantumProbDistOptimized(qc)
    else:
        pd = QuantumProbDist(qc)
    return pd

def uniform_discrete(rvc):
    rand = np.random.rand(*rvc.outcome_space.get_input_base())
    normed = rand / np.sum(rand)
    return ProbDist(rvc, normed)

def c4_type(rvc):
    rand = np.random.randint(2, size=rvc.outcome_space.get_input_base())
    normed = rand / np.sum(rand)
    return ProbDist(rvc, normed)

def spekkens(rvc):
    perm123 = list(itertools.permutations([1,2,3]))
    rest = list(utils.flatten([utils.unique_everseen(itertools.permutations([0,i,i])) for i in range(4)]))
    allowed_outcomes = perm123 + rest
    support = np.zeros((4,4,4))
    p = 1/len(allowed_outcomes)
    for i in allowed_outcomes:
        support[i] = p
    # print(support)
    return ProbDist(rvc, support)

def tsirelson(rvc):

    def p(a,b,x,y):
        xy = 1 if not (x and y) else -1
        a_ = 1 if a else -1
        b_ = 1 if b else -1
        return 1/16 * (1 + (xy * a_ * b_)/sqrt2)

    return ProbDist.from_callable_support(rvc, p)

def fritz(rvc):
    ei = utils.ei
    pi = np.pi
    perm = utils.get_triangle_permutation()
    # Eigenvectors of sigma_x
    e_x_0 = (qb0 + qb1)/(sqrt2)
    e_x_1 = (-qb0 + qb1)/(sqrt2)
    # Eigenvectors of sigma_y
    e_y_0 = (i*qb0 + qb1)/(sqrt2)
    e_y_1 = (-i*qb0 + qb1)/(sqrt2)
    # Eigenvectors of -(sigma_y + sigma_x)/sqrt2
    e_yx_0 = (ei(-3/4*pi)*qb0 + qb1)/(sqrt2)
    e_yx_1 = (ei(1/4*pi)*qb0 + qb1)/(sqrt2)
    # Eigenvectors of (sigma_y - sigma_x)/sqrt2
    e_xy_0 = (ei(-1/4*pi)*qb0 + qb1)/(sqrt2)
    e_xy_1 = (ei(-5/4*pi)*qb0 + qb1)/(sqrt2)

    rho0 = utils.ket_to_dm(qb0)
    rho1 = utils.ket_to_dm(qb1)
    # phi0 = utils.ket_to_dm(e3)
    # phi1 = utils.ket_to_dm(e4)
    # omega0 = utils.ket_to_dm(e5)
    # omega1 = utils.ket_to_dm(e6)
    A_measurements = [
        utils.tensor(rho1, utils.ket_to_dm(e_y_0)),
        utils.tensor(rho1, utils.ket_to_dm(e_y_1)),
        utils.tensor(rho0, utils.ket_to_dm(e_x_1)),
        utils.tensor(rho0, utils.ket_to_dm(e_x_0)),
    ]
    B_measurements = [
        utils.tensor(utils.ket_to_dm(e_xy_0), rho0),
        utils.tensor(utils.ket_to_dm(e_xy_1), rho0),
        utils.tensor(utils.ket_to_dm(e_yx_0), rho1),
        utils.tensor(utils.ket_to_dm(e_yx_1), rho1),
    ]
    C_measurements = [
        utils.tensor(rho0, rho1),
        utils.tensor(rho0, rho0),
        utils.tensor(rho1, rho1),
        utils.tensor(rho1, rho0),
    ]

    A = Measurement(A_measurements)
    B = Measurement(B_measurements)
    C = Measurement(C_measurements)
    rhoAB = State.Strats.Deterministic.maximally_entangled_bell(3)
    rhoBC = State.Strats.Deterministic.maximally_entangled_bell(0)
    rhoAC = State.Strats.Deterministic.maximally_entangled_bell(0)

    # js = utils.tensor(rhoAB.data, rhoBC.data, rhoAC.data)
    # print(multidot(rhoAB.data))
    # jm = utils.tensor(utils.tensor(rho1, omega0), utils.tensor(rho1, omega0), utils.tensor(rho0, rho0))
    # t = utils.multidot(perm.T, js, perm, jm)
    qc = QuantumContext(random_variables=rvc, measurements=(A,B,C), states=(rhoAB, rhoBC, rhoAC), permutation=perm)
    pd = QuantumProbDist(qc)
    pd.update_correlation_settings({'method': 'same', 'mod': 2})
    return pd

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
    # spd = spekkens()
    # print(fpd.prob({'B': 0, 'C': 3}))
    # print(fpd.prob({'C': 1, 'A': 0, 'B': 0}))
    # print(fpd._dev_slice((0,0,1)))
    # return
    # print(spd)
    # npd = null()
    # print(npd)
    # print(spd * npd)
    # print(spd.H('B', 'B'))
    # print(spd.I(['B', 'B'], ['B']))
    # return
    pd = demo_distro()
    print(pd)
    print(pd.marginal(['A', 'B']))
    print(pd.marginal(['C']))
    print(pd.marginal(['C']) * pd.marginal(['A', 'B']))
    # print(pd.condition({'A': 'a'}))
    # print(pd.prob({'A': 'a', 'B': 'b1'}))
    # print(pd.prob({'A': 'a', 'C': 'c2'}))

if __name__ == '__main__':
    # import cProfile
    # stats = cProfile.run('perform_tests()')
    # print(stats)
    perform_tests()