from ..contexts.param_context import *
from ..contexts.quantum_context import *
from ..contexts.measurement import *
from ..contexts.state import *
from ..rmt.unitary_param import *
import numpy as np

class ParamCaller(ParamContext):

    def __init__(self, desc):
        super().__init__(desc)

    def context(self, param):
        raise NotImplemented

    def __call__(self, param):
        assert(len(param) == self.size), "Number of parameters ({}) doesn't match size ({}).".format(len(param), self.size)
        context = self.context(param)
        return self._call(context)

    def _call(self, context):
        raise NotImplemented

class QuantumCaller(ParamCaller):

    def __init__(self, target, rvc, perm=None):
        super().__init__([16,16,16,15,15,15])
        self.m = MeasurementParam(4)
        self.s = StateParam(4, 4)
        self.rvc = rvc
        self.perm = perm
        self._target = target

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

    def _call(self, context):
        return self._target(context)