from functools import reduce
import numpy as np
from scipy import optimize, linalg
from ineqs_loader import get_ineq
from utils import *
from quantum_entity_generator import *
from pprint import pprint
from time import sleep
import itertools
from job_queuer_async import JobContext
# from job_queuer import JobContext
from timing_profiler import timing
from operator import attrgetter, itemgetter
# from fast_dot import dot
# from sklearn.preprocessing import normalize
# from qutip import Qobj, tensor, ket2dm, basis, sigmax, sigmay, sigmaz, expect, qeye, qstate

# === Configure ===
np.set_printoptions(precision=3, linewidth=120, suppress=True)

class ParameterizationStrat():

    def __init__(self, name, mem_loc, impl, impl_args=None):
        self.name = name
        self.mem_loc = mem_loc
        self.impl = impl
        self.impl_args = impl_args
        self.mem_slots = gen_memory_slots(self.mem_loc)
        self.mem_size = sum(self.mem_loc)

    def get_context(self, param):
        if self.impl_args:
            return self.impl(self.mem_slots, param, *self.impl_args)
        else:
            return self.impl(self.mem_slots, param)

    @staticmethod
    def maximally_entangled(mem_slots, param, *args):
        p_mA, p_mB, p_mC = mem_slots

        A = get_correl_meas(get_psd_herm_neig(param[p_mA], 4))
        B = get_correl_meas(get_psd_herm_neig(param[p_mB], 4))
        C = get_correl_meas(get_psd_herm_neig(param[p_mC], 4))

        X = get_maximally_entangled_bell_state(args[0])
        Y = get_maximally_entangled_bell_state(args[1])
        Z = get_maximally_entangled_bell_state(args[2])

        return A, B, C, X, Y, Z

    @staticmethod
    def most_general(mem_slots, param, *args):
        p_mA, p_mB, p_mC, p_X, p_Y, p_Z = mem_slots

        A = get_correl_meas(get_psd_herm_neig(param[p_mA], 4))
        B = get_correl_meas(get_psd_herm_neig(param[p_mB], 4))
        C = get_correl_meas(get_psd_herm_neig(param[p_mC], 4))

        X = get_psd_herm_ntr(param[p_X], 4)
        Y = get_psd_herm_ntr(param[p_Y], 4)
        Z = get_psd_herm_ntr(param[p_Z], 4)

        return A, B, C, X, Y, Z

    @staticmethod
    def pure_pure(mem_slots, param, *args):
        p_mA, p_mB, p_mC, p_X, p_Y, p_Z = mem_slots

        A = get_correl_meas(get_tqds_dm(param[p_mA]))
        B = get_correl_meas(get_tqds_dm(param[p_mB]))
        C = get_correl_meas(get_tqds_dm(param[p_mC]))

        X = get_tqds_dm(param[p_mA])
        Y = get_tqds_dm(param[p_mB])
        Z = get_tqds_dm(param[p_mC])

        return A, B, C, X, Y, Z

# === Configure ===
OPT_STRAT_CONFIG = [
    ParameterizationStrat(
        name='maximally_entangled',
        mem_loc=[16,16,16],
        impl=ParameterizationStrat.maximally_entangled,
        impl_args=(0,0,0),
    ),
    ParameterizationStrat(
        name='most_general',
        mem_loc=[16,16,16,16,16,16],
        impl=ParameterizationStrat.most_general,
    ),
    ParameterizationStrat(
        name='pure_pure',
        mem_loc=[1,1,1,1,1,1],
        impl=ParameterizationStrat.pure_pure,
    )
]

class TriangleUtils():

    @staticmethod
    def explode_correlator(c):
        c_list = [
            c['_'],
            c['A'],
            c['B'],
            c['C'],
            c['AB'],
            c['AC'],
            c['BC'],
            c['ABC'],
            c['A']*c['B'],
            c['A']*c['C'],
            c['B']*c['C'],
            c['C']*c['AB'],
            c['B']*c['AC'],
            c['A']*c['BC'],
            c['A']*c['B']*c['C'],
        ]
        return c_list

    @staticmethod
    def gen_correlator(diffM, state):
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

class Correlation_Minimizer():

    def __init__(self, ineq_index, strat_index, target_value = 0, local_log = False):
        self.__solved__ = False
        self.best_objective_result = np.inf
        self.best_objective_result_param = None
        self.ineq_index = ineq_index
        self.ineq_coeff = get_ineq(ineq_index)
        self.string_logs = []
        self.target_value = target_value
        self.strat_index = strat_index
        self.strat = OPT_STRAT_CONFIG[strat_index]
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

    def minimize_callback(self, param):
        objective_res = self.objective(param)
        self.log("Step result: {0}", objective_res)

    def basin_hopping_callback(self, param, objective_res, accept):
        self.log("Step result: {0}", objective_res)
        self.log("Local Minimum Accepted?: {0}", accept)
        if (objective_res < self.best_objective_result):
            self.best_objective_result = objective_res
            self.best_objective_result_param = param
            self.log("New Best Objective Result: {0}", objective_res)
        should_terminate = (objective_res <= self.target_value)
        return should_terminate

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
            raise Exception("Already Solved")
        # initial_guess = 50 * (2*np.random.random(self.strat.mem_size) - 1)
        # initial_guess = 50 * (2*np.random.random(self.strat.mem_size) - 1)
        initial_guess = np.random.normal(scale=1.0, size=self.strat.mem_size)
        self.log("Minimizing")
        res = optimize.minimize(
            fun = self.objective,
            x0 = initial_guess,
            callback = self.minimize_callback,
            # tol = 1e-3,
            method='L-BFGS-B',
            # method='BFGS',
            # method = 'SLSQP',
            # options = {
            #     'maxiter':40,
            # },
        )
        self.__solved__ = True
        self.log("Solved")
        self.best_objective_result_param = res.x
        self.best_objective_result = self.objective(self.solution)
        self.context = self.strat.get_context(self.solution)

    def basin_hopping(self):
        if self.__solved__:
            raise Exception("Already Solved")

        initial_guess = np.random.normal(scale=1.0, size=self.strat.mem_size)
        # initial_guess = np.zeros(self.strat.mem_size)
        res = optimize.basinhopping(
            func = self.objective,
            x0 = initial_guess,
            callback = self.basin_hopping_callback,
            # T = 1e-2,
            stepsize = 1e2,
            minimizer_kwargs = {
                'method': 'L-BFGS-B',
                'tol': 1e-4,
            },
        )
        self.__solved__ = True
        self.log("Solved")
        self.context = self.strat.get_context(self.best_objective_result_param)

    def get_exploded_correlator(self, param):
        A, B, C, X, Y, Z = self.strat.get_context(param)

        XxYxZ = tensor(X, Y, Z)
        # print(XxYxZ)
        state = reduce(np.dot, (self.perm.T, XxYxZ, self.perm))
        # print(state)

        diffM = (A, B, C)
        c = TriangleUtils.gen_correlator(diffM, state)

        c_exploded = TriangleUtils.explode_correlator(c)
        return c_exploded

    def objective(self, param):
        c_exploded = self.get_exploded_correlator(param)
        target_value = np.dot(self.ineq_coeff, c_exploded)
        # print(target_value)
        return target_value

def find_max_violation(ineq_index):
    # num_trys_per_job = 10
    # Create instance, eval, organize output into serial
    # cm_dict_list = []
    # for i in range(num_trys_per_job):
    cm = Correlation_Minimizer(
        ineq_index=ineq_index,
        strat_index=1,
        local_log=False,
        target_value=5e-2
    )
    # cm.minimize()
    cm.basin_hopping()
    cm_dict = {
        'solution': cm.best_objective_result_param,
        'ineq_index': cm.ineq_index,
        'objective_result': cm.best_objective_result,
        'context': cm.context,
        'logs': cm.string_logs,
        'correlator': cm.get_exploded_correlator(cm.best_objective_result_param),
    }
        # cm_dict_list.append(cm_dict)
        # if cm.objective_result <= target_value:
        #     break
    # best_cm_dict = min(cm_dict_list, key=itemgetter('objective_result'))

    # explore_best(best_cm_dict)
    # return best_cm_dict
    return cm_dict

# def explore_best(best_cm_dict):
#     context = best_cm_dict['context']
#     A, B, C, X, Y, Z = context
#     XYZ = tensor(X, Y, Z)
#     print(XYZ)
#     print(np.trace(XYZ))

@timing
def main():
    # perm = get_perumation()
    # ident = reduce(np.dot, (perm,perm,perm,perm,perm,perm))
    # print(ident)
    # return
    # print(get_correl_meas(get_tqds_dm(np.pi / 2)))
    # return

    # pprint(find_max_violation(8 - 1))
    # find_max_violation(8 - 1)
    # return

    jc = JobContext(
        target_func=find_max_violation,
        target_args=[[i] for i in range(38)],
        log_worker=True,
        num_cores=-1,
    )
    print("Evaluating...")
    jc.evaluate()
    results_array = []
    saturated = []
    for result in jc.target_results:
        ids = np.asarray([result['ineq_index'], result['objective_result']])
        results_array.append(np.hstack((ids, result['solution'])))
        if result['objective_result'] < 5e-2:
            saturated.append(int(result['ineq_index']))
    results_array = np.asarray(results_array)
    np.savetxt('solutions/temp.csv', results_array, header='ineq_coeff, obj_result, params...')
    print("Results Saved.")
    print("Saturated inequalities:")
    saturated = np.asarray(saturated) + 1
    print(np.sort(saturated))

if __name__ == '__main__':
    main()
