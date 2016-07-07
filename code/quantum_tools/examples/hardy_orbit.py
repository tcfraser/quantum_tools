import numpy as np
from ..optimizers.minimizer import Minimizer
from ..utilities import utils
from ..config import *
from ..contexts.measurement import Measurement
from ..contexts.state import State
from ..statistics.variable import RandomVariableCollection
from ..inflation import marginal_equality
from ..examples import symbolic_contexts
from ..contexts.quantum_context import QuantumContext, QuantumProbDist

class HardyOrbitMinimizer(Minimizer):

    def __init__(self, config):
        Minimizer.__init__(self, [32,32,32,16,16,16])
        self.local_log = True
        self.permutation = utils.get_triangle_permutation()
        self.random_variables = RandomVariableCollection.new(('A', 'B', 'C'), (4, 4, 4))
        self.preinjectable_sets = symbolic_contexts.ABC_444_444.preinjectable_sets
        self.orbit_contractor = utils.load_sparse("ABC_444_444_row_sum.mtx")
        self.antecedent = 0
        self,

    def initial_guess(self):
        initial_guess = np.random.normal(scale=10.0, size=self.mem_size)
        return initial_guess

    def get_context(self, param):
        pA, pB, pC, prhoAB, prhoBC, prhoAC = self.mem_slots
        A = Measurement.Strats.Param.pvms(param[pA])
        B = Measurement.Strats.Param.pvms(param[pB])
        C = Measurement.Strats.Param.pvms(param[pC])
        rhoAB = State.Strats.Param.dm(param[prhoAB])
        rhoBC = State.Strats.Param.dm(param[prhoBC])
        rhoAC = State.Strats.Param.dm(param[prhoAC])

        qc = QuantumContext(
            random_variables=self.random_variables,
            measurements=(A,B,C),
            states=(rhoAB,rhoBC,rhoAC),
            permutation=self.permutation,
        )
        return qc

    def objective(self, param):
        qc = self.get_context(param)
        pd = QuantumProbDist(qc)

        # target = pd._dev_slice((0,0,0))
        b = marginal_equality.contexts_marginals(pd, self.preinjectable_sets)
        orbit_sum = self.orbit_contractor.dot(b)
        target = orbit_sum[0]
        self.log("Calculated objective", target)
        return target

def go():
    hom = HardyOrbitMinimizer()
    hom.minimize()
    hom.save_results_to_file(OUTPUT_DIR + "HOM_temp.txt")

if __name__ == '__main__':
    PROFILE_MIXIN(go)
