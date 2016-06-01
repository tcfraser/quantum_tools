from __future__ import print_function, division
import numpy as np
from minimizer import Minimizer
from utils import Utils
import global_config
from measurement import Measurement
from state import State
from quantum_pd import QuantumContext, QuantumProbDistro
from ineqs import *

class TEM(Minimizer):

    def __init__(self):
        Minimizer.__init__(self, [16,16,16,16,16,16])
        self.local_log=True
        self.permutation = Utils.get_permutation()

    def initial_guess(self,):
        # d = [1.0]*4 + [0.0]*12
        # initial_guess = list(np.random.normal(scale=1.0, size=16*3)) + d + d + d
        initial_guess = np.random.normal(scale=10.0, size=self.mem_size)
        # print(initial_guess)
        return initial_guess

    def get_context(self, param):
        pA, pB, pC, prhoAB, prhoBC, prhoAC = self.mem_slots
        A = Measurement.pvms('A', param[pA])
        B = Measurement.pvms('B', param[pB])
        C = Measurement.pvms('C', param[pC])
        rhoAB = State.dm(param[prhoAB])
        rhoBC = State.dm(param[prhoBC])
        rhoAC = State.dm(param[prhoAC])

        qc = QuantumContext(measurements=(A,B,C), states=(rhoAB,rhoBC,rhoAC), permutation=self.permutation)
        return qc

    def get_prob_distrobution(self, context):
        return QuantumProbDistro(context)

    def objective(self, param):
        qc = self.get_context(param)
        pd = self.get_prob_distrobution(qc)

        # IAB = pd.mutual_information(0,1)
        # IAC = pd.mutual_information(0,2)
        # HA = pd.entropy((1,2))

        # target = HA - IAB - IAC
        # print(IAB, IAC, HA, target)
        target = HLP3(pd)
        print(target)
        return target

def run_minimizer():
    tem = TEM()
    tem.minimize()
    tem.save_results_to_file("TEM")

if __name__ == '__main__':
    run_minimizer()
