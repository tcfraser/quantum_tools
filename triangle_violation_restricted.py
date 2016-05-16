from functools import reduce
import numpy as np
from scipy import optimize, linalg
from ineqs_loader import get_ineq
from utils import *
from pprint import pprint
from time import sleep
import itertools
from job_queuer import JobContext
from timing_profiler import timing
# from sklearn.preprocessing import normalize
# from qutip import Qobj, tensor, ket2dm, basis, sigmax, sigmay, sigmaz, expect, qeye, qstate

# === Configure ===
np.set_printoptions(precision=3, linewidth=120, suppress=True)
TOLERANCE = 1e-4
# maximally_entangled = (1,2,3)
maximally_entangled = False
if maximally_entangled:
    MEM_LOC = [1,16,1,16,1,16]
else:
    MEM_LOC = [1,16,1,16,1,16,16,16,16]


class Correlation_Minimizer():

    def __init__(self, ineq_index, local_log = False):
        self.__solved__ = False
        self.ineq_index = ineq_index
        self.ineq_coeff = get_ineq(ineq_index)
        self.string_logs = []
        self.mem_loc = MEM_LOC
        self.local_log = local_log
        self.log('Initialized')
        self.log('ineq_index: {0}', ineq_index)
        self.log(self.gen_ineq_string())
        self.perm = get_perumation()

    def log(self, log_item, *args):
        if isinstance(log_item, str) and args is not None:
            log_item = log_item.format(*args)
        if self.local_log:
            pprint(log_item)
        self.string_logs.append([log_item])

    def after_iterate(self, param):
        objective_res = self.objective(param)
        self.log("Step result: {0}", objective_res)

    def gen_ineq_string(self, result=None):
        human_readable = ["", "<A>", "<B>", "<C>", "<AB>", "<AC>", "<BC>", "<ABC>", "<A><B>", "<A><C>", "<B><C>", "<C><AB>", "<B><AC>", "<A><BC>", "<A><B><C>"]
        ineq_str = ''
        for i, coeff in enumerate(self.ineq_coeff):
            if coeff != 0:
                segment = '{0:+g}'.format(coeff) + human_readable[i]
                if (i > 0):
                    segment = segment.replace("1", "")
                ineq_str += segment
        if result:
            ineq_str = str(result) + " = " + ineq_str
        ineq_str = "0 <= " + ineq_str
        return ineq_str

    def minimize(self):
        if self.__solved__:
            return
        initial_guess = np.random.normal(scale=2.0, size=(sum(self.mem_loc)))
        self.log("Minimizing")
        res = optimize.minimize(
            fun = self.objective,
            x0 = initial_guess,
            callback = self.after_iterate,
            tol = TOLERANCE,
        )
        self.__solved__ = True
        self.solution = res.x
        self.objective_result = self.objective(self.solution)
        self.context = self.get_context(self.solution)

    def get_exploded_correlator(self, param):
        dA, dB, dC, sX, sY, sZ = self.get_context(param)

        XxYxZ = tensor(sX, sY, sZ)
        state = reduce(np.dot, (self.perm.T, XxYxZ, self.perm))
        diffM = (dA, dB, dC)
        c = self.gen_correlator(diffM, state)

        c_exploded = self.explode_correlator(c)
        return c_exploded

    def objective(self, param):
        c_exploded = self.get_exploded_correlator(param)
        target_value = np.dot(self.ineq_coeff, c_exploded)
        return target_value

    def get_context(self, param):
        # print(p_wA, p_mA, p_wB, p_mB, p_wC, p_mC, p_sX, p_sY, p_sZ)
        # print(param)
        if maximally_entangled:
            p_wA, p_mA, p_wB, p_mB, p_wC, p_mC = gen_memory_slots(self.mem_loc)
        else:
            p_wA, p_mA, p_wB, p_mB, p_wC, p_mC, p_sX, p_sY, p_sZ = gen_memory_slots(self.mem_loc)

        wA = param[p_wA][0]
        mA = param[p_mA]
        wB = param[p_wB][0]
        mB = param[p_mB]
        wC = param[p_wC][0]
        mC = param[p_mC]

        dA = self.get_meas(wA, mA)
        dB = self.get_meas(wB, mB)
        dC = self.get_meas(wC, mC)

        if maximally_entangled:
            sX = get_maximally_entangled_bell_state(maximally_entangled[0])
            sY = get_maximally_entangled_bell_state(maximally_entangled[1])
            sZ = get_maximally_entangled_bell_state(maximally_entangled[2])
        else:
            sX = get_psd_herm(p_sX, 4)
            sY = get_psd_herm(p_sY, 4)
            sZ = get_psd_herm(p_sZ, 4)

        return dA, dB, dC, sX, sY, sZ

    def get_meas(self, weight, param):
        M_y = norm_real_parameter(weight) * get_psd_herm(param, 4, unitary_trace=True)
        M_n = I4 - M_y
        dM = M_y - M_n
        return dM

    def gen_correlator(self, diffM, state):
        c = {}
        m = len(diffM)
        for i in itertools.product(*[[0,1]]*m):
            label = ''
            measures = []
            for index, present in enumerate(i):
                if present:
                    label += chr(index + ord('A'))
                    measures.append(diffM[index])
                else:
                    measures.append(I4)
            if label == '':
                label = '_'
            joint_measure = tensor(*measures)
            correl = np.trace(np.dot(state, joint_measure))
            # if not is_close(correl.imag, 0):
            #     print(correl.imag)
            # assert(is_close(correl.imag, 0)), "Correlation ({0}) are not real; check hermitianicity.".format(correl)
            c[label] = correl.real
        return c

    def explode_correlator(self, c):
        c_list = []
        c_list.append(c['_'])
        c_list.append(c['A'])
        c_list.append(c['B'])
        c_list.append(c['C'])
        c_list.append(c['AB'])
        c_list.append(c['AC'])
        c_list.append(c['BC'])
        c_list.append(c['ABC'])
        c_list.append(c['A']*c['B'])
        c_list.append(c['A']*c['C'])
        c_list.append(c['B']*c['C'])
        c_list.append(c['C']*c['AB'])
        c_list.append(c['B']*c['AC'])
        c_list.append(c['A']*c['BC'])
        c_list.append(c['A']*c['B']*c['C'])
        return c_list

def find_max_violation(ineq_index):
    # Create instance, eval, organize output into serial
    cm = Correlation_Minimizer(ineq_index, local_log=False)
    cm.minimize()
    cm_dict = {
        'solution': cm.solution,
        'ineq_index': cm.ineq_index,
        'objective_result': cm.objective_result,
        'context': cm.context,
        'logs': cm.string_logs,
        'correlator': cm.get_exploded_correlator(cm.solution),
    }

    return cm_dict

@timing
def main():
    # pprint(find_max_violation(0))
    # return

    jc = JobContext(find_max_violation, [[i] for i in range(38)], log_worker=True)
    print("Evaluating...")
    jc.evaluate()
    results_array = []
    saturated = []
    for result in jc.target_results:
        ids = np.asarray([result['ineq_index'], result['objective_result']])
        results_array.append(np.hstack((ids, result['solution'])))
        if result['objective_result'] < 1e-3:
            saturated.append(int(result['ineq_index']))
    results_array = np.asarray(results_array)
    np.savetxt('solutions/restricted.csv', results_array, header='ineq_coeff, obj_result, params...')
    print("Results Saved.")
    print("Saturated inequalities:")
    saturated = np.asarray(saturated) + 1
    print(np.sort(saturated))

if __name__ == '__main__':
    main()
