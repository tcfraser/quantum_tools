from __future__ import print_function, division
import numpy as np
from ..optimizers.minimizer import Minimizer
from ..utilities import utils
from ..utilities import rmt
from ..config import *
from ..contexts.measurement import Measurement
from ..contexts.state import State
from ..contexts.quantum_context import QuantumContext, QuantumProbDist
from .ineqs_loader import *

class INEQ_Triangle(Minimizer):

    def __init__(self):
        Minimizer.__init__(self, [32,32,32,16,16,16])
        self.local_log=True
        self.permutation = utils.get_permutation()
        self.seed_operator_A = rmt.P_I(4,2)
        self.seed_operator_B = rmt.P_I(4,2)
        self.seed_operator_C = rmt.P_I(4,2)
        self.ineq = get_ineq(8 - 1)

    def initial_guess(self,):
        # d = [1.0]*4 + [0.0]*12
        # initial_guess = list(np.random.normal(scale=1.0, size=16*3)) + d + d + d
        initial_guess = np.random.normal(scale=1.0, size=self.mem_size)
        # print(initial_guess)
        return initial_guess

    def get_context(self, param):
        pA, pB, pC, prhoAB, prhoBC, prhoAC = self.mem_slots
        A = Measurement.Strats.Param.pvms('A', param[pA], self.seed_operator_A)
        B = Measurement.Strats.Param.pvms('B', param[pB], self.seed_operator_B)
        C = Measurement.Strats.Param.pvms('C', param[pC], self.seed_operator_C)
        rhoAB = State.dm(param[prhoAB])
        rhoBC = State.dm(param[prhoBC])
        rhoAC = State.dm(param[prhoAC])

        qc = QuantumContext(measurements=(A,B,C), states=(rhoAB,rhoBC,rhoAC), permutation=self.permutation)
        return qc

    def get_prob_distribution(self, context):
        return QuantumProbDist(context)

    def objective(self, param):
        qc = self.get_context(param)
        pd = self.get_prob_distribution(qc)

        labels = ['A', 'B', 'C']
        C = {}
        for i in utils.powerset(labels):
            C[''.join(i)] = pd.coincidence(i,method='two_expect')
        C_list = [
            C[''],
            C['A'],
            C['B'],
            C['C'],
            C['AB'],
            C['AC'],
            C['BC'],
            C['ABC'],
            C['A']*C['B'],
            C['A']*C['C'],
            C['B']*C['C'],
            C['C']*C['AB'],
            C['B']*C['AC'],
            C['A']*C['BC'],
            C['A']*C['B']*C['C'],
        ]
        # print(C_list)
        target = np.dot(C_list, self.ineq)
        # print(self.ineq)
        # print(C_list)
        print(target)
        # assert(False)
        return target

def run_minimizer():
    m = INEQ_Triangle()
    m.minimize()
    m.save_results_to_file(OUTPUT_DIR + "INEQ_Triangle_temp.txt")

if __name__ == '__main__':
    run_minimizer()
