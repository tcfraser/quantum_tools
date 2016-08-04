from ..contexts.param_context import *
from ..contexts.quantum_context import *
from ..contexts.measurement import *
from ..contexts.state import *
from ..rmt.unitary_param import *
import numpy as np

class ParamCaller(ParamContext):

    def __init__(self, target, desc):
        super().__init__(desc)
        self._target = target

    def context(self, param):
        raise NotImplemented

    def __call__(self, param):
        assert(len(param) == self.size), "Number of parameters ({}) doesn't match size ({}).".format(len(param), self.size)
        context = self.context(param)
        return self._call(context)

    def _call(self, context):
        return self._target(context)

class ConvexityCaller(ParamCaller):
    
    def __init__(self, target, rvc):
        super().__init__(target, [len(rvc.outcome_space)])
        self.rvc = rvc
        
    def context(self, param):
        normed_params = param**2
        normed_params /= np.sum(normed_params)
        support = normed_params.reshape(self.rvc.outcome_space.get_input_base())
        
        return ProbDist(self.rvc, support)
        
class QuantumCaller(ParamCaller):

    def __init__(self, target, rvc, perm=None):
        super().__init__(target, [16,16,16,15,15,15])
        self.m = MeasurementParam(4)
        self.s = StateParam(4, 4)
        self.rvc = rvc
        self.perm = perm

    def q_context(self, param):
        sA, sB, sC, srhoAB, srhoBC, srhoCA = self.slots

        A = ProjectiveMeasurement(self.m.gen(param[sA]))
        B = ProjectiveMeasurement(self.m.gen(param[sB]))
        C = ProjectiveMeasurement(self.m.gen(param[sC]))
        rhoAB = State(self.s.gen(param[srhoAB]))
        rhoBC = State(self.s.gen(param[srhoBC]))
        rhoCA = State(self.s.gen(param[srhoCA]))

        qc = QuantumContext(
            random_variables=self.rvc,
            measurements=(A,B,C),
            states=(rhoAB,rhoBC,rhoCA),
            permutation=self.perm,
        )
        return qc

    def context(self, param):
        qc = self.q_context(param)
        pd = QuantumProbDistOptimized(qc)

        return pd
    
class QuantumCaller(ParamCaller):

    def __init__(self, target, rvc, perm=None):
        super().__init__(target, [16,16,16,15,15,15])
        self.m = MeasurementParam(4)
        self.s = StateParam(4, 4)
        self.rvc = rvc
        self.perm = perm

    def q_context(self, param):
        sA, sB, sC, srhoAB, srhoBC, srhoCA = self.slots

        A = ProjectiveMeasurement(self.m.gen(param[sA]))
        B = ProjectiveMeasurement(self.m.gen(param[sB]))
        C = ProjectiveMeasurement(self.m.gen(param[sC]))
        rhoAB = State(self.s.gen(param[srhoAB]))
        rhoBC = State(self.s.gen(param[srhoBC]))
        rhoCA = State(self.s.gen(param[srhoCA]))

        qc = QuantumContext(
            random_variables=self.rvc,
            measurements=(A,B,C),
            states=(rhoAB,rhoBC,rhoCA),
            permutation=self.perm,
        )
        return qc

    def context(self, param):
        qc = self.q_context(param)
        pd = QuantumProbDistOptimized(qc)

        return pd
    
class StateRestrictedQuantumCaller(ParamCaller):

    def __init__(self, target, rvc, perm=None, states=(0,0,0)):
        super().__init__(target, [16,16,16])
        self.m = MeasurementParam(4)
        self.rvc = rvc
        self.perm = perm
        self.states = states

    def q_context(self, param):
        sA, sB, sC = self.slots

        A = ProjectiveMeasurement(self.m.gen(param[sA]))
        B = ProjectiveMeasurement(self.m.gen(param[sB]))
        C = ProjectiveMeasurement(self.m.gen(param[sC]))
        
        rhoAB = State.Strats.Deterministic.maximally_entangled_bell(self.states[0])
        rhoBC = State.Strats.Deterministic.maximally_entangled_bell(self.states[1])
        rhoCA = State.Strats.Deterministic.maximally_entangled_bell(self.states[2])

        qc = QuantumContext(
            random_variables=self.rvc,
            measurements=(A,B,C),
            states=(rhoAB,rhoBC,rhoCA),
            permutation=self.perm,
        )
        return qc

    def context(self, param):
        qc = self.q_context(param)
        pd = QuantumProbDistOptimized(qc)

        return pd
    
class SymmetricStateSymmetricMeasurementQuantumCaller(ParamCaller):

    def __init__(self, target, rvc, perm=None):
        super().__init__(target, [16,15])
        self.m = MeasurementParam(4)
        self.s = StateParam(4, 4)
        self.rvc = rvc
        self.perm = perm

    def q_context(self, param):
        sM, sS = self.slots

        A = ProjectiveMeasurement(self.m.gen(param[sM]))
        B = C = A
        
        rhoAB = State(self.s.gen(param[sS]))
        rhoBC = rhoCA = rhoAB

        qc = QuantumContext(
            random_variables=self.rvc,
            measurements=(A,B,C),
            states=(rhoAB,rhoBC,rhoCA),
            permutation=self.perm,
        )
        return qc

    def context(self, param):
        qc = self.q_context(param)
        pd = QuantumProbDistOptimized(qc)

        return pd
    
class LocalUnitaryQuantumCaller(ParamCaller):

    def __init__(self, target, rvc, perm=None):
        super().__init__(target, [4,4,4,4,4,4])
        self.lu = UnitaryParam(2)
        self.rvc = rvc
        self.perm = perm

    def q_context(self, param):
        # sAlu1, sAlu2, sBlu1, sBlu2, sClu1, sClu2 = self.slots
        
        lus = [self.lu.gen(param[lups]) for lups in self.slots]
        
        num_measures = len(lus)//2
        glus = [utils.tensor(lus[i], lus[i+num_measures]) for i in range(num_measures)]
        
        basis = np.eye(4)
        A = ProjectiveMeasurement.from_cols(glus[0].dot(basis))
        B = ProjectiveMeasurement.from_cols(glus[1].dot(basis))
        C = ProjectiveMeasurement.from_cols(glus[2].dot(basis))
        
        rhoAB = State.Strats.Deterministic.maximally_entangled_bell(0)
        rhoBC = State.Strats.Deterministic.maximally_entangled_bell(0)
        rhoCA = State.Strats.Deterministic.maximally_entangled_bell(0)

        qc = QuantumContext(
            random_variables=self.rvc,
            measurements=(A,B,C),
            states=(rhoAB,rhoBC,rhoCA),
            permutation=self.perm,
        )
        return qc

    def context(self, param):
        qc = self.q_context(param)
        pd = QuantumProbDistOptimized(qc)

        return pd
    