import numpy as np
from minimizer import Minimizer
from utils import Utils
import global_config
from measurement import Measurement
from state import State
from quantum_pd import QuantumContext, QuantumProbDistro

class TEM(Minimizer):

    def __init__(self):
        super().__init__([6,6,6])
        self.local_log=True
        self.permutation = Utils.get_permutation()

    def initial_guess(self,):
        return np.random.normal(scale=10.0, size=self.mem_size)

    def get_context(self, param):
        pA, pB, pC = self.mem_slots
        A = Measurement.sbs(param[pA])
        B = Measurement.sbs(param[pB])
        C = Measurement.sbs(param[pC])
        rhoAB = State.mebs(2)
        rhoBC = State.mebs(0)
        rhoAC = State.mebs(0)

        qc = QuantumContext(measurements=(A,B,C), states=(rhoAB,rhoBC,rhoAC), permutation=self.permutation)
        return qc

    def get_prob_distrobution(self, context):
        return QuantumProbDistro(context)

    def objective(self, param):
        qc = self.get_context(param)
        pd = self.get_prob_distrobution(qc)

        IAB = pd.mutual_information(0,1)
        IAC = pd.mutual_information(0,2)
        HA = pd.entropy(0)

        target = HA - IAB - IAC
        # print(IAB, IAC, HA, target)
        return target

def run_minimizer():
    tem = TEM()
    tem.minimize()

if __name__ == '__main__':
    run_minimizer()
