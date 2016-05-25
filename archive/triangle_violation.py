from functools import reduce
import numpy as np
from scipy import optimize, linalg
from ineqs_loader import get_ineqs
from utils import *
import itertools
from triangle_violation_restricted import TriangleUtils
# from sklearn.preprocessing import normalize
# from qutip import Qobj, tensor, ket2dm, basis, sigmax, sigmay, sigmaz, expect, qeye, qstate

# === Configure ===
np.set_printoptions(precision=3, linewidth=120, suppress=True)

# === Constants ===
ineq_coeff = None
perm = get_perumation()

# def get_projective_meas(weight, param):
#     # weight = min(max(weight, 0), 1)
#     assert (len(param) == 7), "Projective state needs 7 parameters."
#     # assert (0 <= weight <= 1), "Weight {w} is not normalized.".format(w=weight)
#     state = get_two_qubit_state(param)
#     M_y = norm_real_parameter(weight) * ket_to_dm(state)
#     M_n = I4 - M_y
#     dM = M_y - M_n
#     return dM

def get_meas(weight, param):
    M_y = weight * get_psd_herm(param, 4, unitary_trace=True)
    M_n = I4 - M_y
    dM = M_y - M_n
    return dM

def get_density_matrix(param):
    return get_psd_herm(param, 4, unitary_trace=True)

# === Memory Locations ===
mem_loc = [1,16,1,16,1,16,16,16,16]
p_wA, p_mA, p_wB, p_mB, p_wC, p_mC, p_sX, p_sY, p_sZ = gen_memory_slots(mem_loc)

def unpack_param(param):
    # print(p_wA, p_mA, p_wB, p_mB, p_wC, p_mC, p_sX, p_sY, p_sZ)
    # print(param)

def get_context(param):

    wA = norm_real_parameter(param[p_wA][0])
    mA = param[p_mA]
    wB = norm_real_parameter(param[p_wB][0])
    mB = param[p_mB]
    wC = norm_real_parameter(param[p_wC][0])
    mC = param[p_mC]
    dA = get_meas(wA, mA)
    dB = get_meas(wB, mB)
    dC = get_meas(wC, mC)

    sX = param[p_sX]
    sY = param[p_sY]
    sZ = param[p_sZ]
    sX = get_density_matrix(sX)
    sY = get_density_matrix(sY)
    sZ = get_density_matrix(sZ)
    return dA, dB, dC, sX, sY, sZ

def objective(param):
    dA, dB, dC, sX, sY, sZ = get_context(param)
    XxYxZ = tensor(sX, sY, sZ)
    state = reduce(np.dot, (perm.T, XxYxZ, perm))
    # state = XxYxZ
    diffM = (dA, dB, dC)
    # assert(is_hermitian(state))
    # for dM in diffM:
    #    assert(is_hermitian(dM))
    # assert(is_hermitian(sX))
    # assert(is_hermitian(sY))
    # assert(is_hermitian(sZ))
    # assert(is_hermitian(dA))
    # assert(is_hermitian(dB))
    # assert(is_hermitian(dC))
    # assert(is_hermitian(state))
    # assert(is_hermitian(XxYxZ))
    c = TriangleUtils.gen_correlator(diffM, state)

    target_value = np.dot(ineq_coeff, TriangleUtils.explode_correlator(c))
    return target_value

def print_ineq_str(result=None):
    human_readable = ["", "<A>", "<B>", "<C>", "<AB>", "<AC>", "<BC>", "<ABC>", "<A><B>", "<A><C>", "<B><C>", "<C><AB>", "<B><AC>", "<A><BC>", "<A><B><C>"]
    ineq_str = "".join(['{0:+g}'.format(i) + j  for (i, j) in zip(ineq_coeff, human_readable)])
    ineq_str = ineq_str.replace("1", "")
    if result:
        ineq_str = str(result) + " = " + ineq_str
    ineq_str = "0 <= " + ineq_str
    print(ineq_str)

def log_results(param):
    print("=== Complete ===")
    print("Results:")
    objective_res = objective(param)
    print_ineq_str(objective_res)
    print("Objective Value:", objective_res)
    A, B, C, X, Y, Z = get_context(param)
    wA, mA, wB, mB, wC, mC, sX, sY, sZ = unpack_param(param)
    print("Measure State A (w={0}):".format(wA))
    print(A)
    print("Measure State B (w={0}):".format(wB))
    print(B)
    print("Measure State C (w={0}):".format(wC))
    print(C)
    print("States rho_X:")
    print(X)
    print("States rho_Y:")
    print(Y)
    print("States rho_Z:")
    print(Z)

    # AxBxC = tensor(A, B, C)
    # XxYxZ


    # print(is_trace_one(sX))
    # print(is_trace_one(sY))
    # print(is_trace_one(sZ))

def after_iterate(param):
    objective_res = objective(param)
    print("Step result:", objective_res)
    if objective_res < -10:
        log_results(param)
        assert(False)

def find_max_violation():
    # initial_guess = np.random.random((sum(mem_loc))) * -1
    initial_guess = np.random.normal(scale=1.0, size=(sum(mem_loc)))

    # print(initial_guess)
    # log_results(initial_guess)
    # return
    print("=== Minimizing ===")
    res = optimize.minimize(
        fun = objective,
        x0 = initial_guess,
        callback = after_iterate,
        tol = 1e-4,
        # args = (ineq_coeff,),
        # method='SLSQP',
        # constraints = [
        #     {
        #         'type': 'ineq',
        #         'fun': bounded_weight,
        #         'args': (0,) # weight_A
        #     },
        #     {
        #         'type': 'ineq',
        #         'fun': bounded_weight,
        #         'args': (1,) # weight_B
        #     },
        #     {
        #         'type': 'ineq',
        #         'fun': bounded_weight,
        #         'args': (2,) # weight_C
        #     },
        # ]
    )
    log_results(res.x)

def main():
    # === Tests ===
    # ex_measurement = get_meas(0.3 , [1,2,3,4,5,6,7])
    # print(ex_measurement)
    # ex_perm = get_perumation()
    # print(ex_perm)
    # ex_state = get_density_matrix([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    # assert(np.array_equal(ex_state, ex_state.conj().T)), "Example not hermitian"
    # assert(np.all(linalg.eigvals(ex_state) > 0)), "Example not positive definite"
    # print(ex_state)
    # param = np.random.random((sum(mem_loc)))
    # objective(param)

    global ineq_coeff
    ineq_coeff = get_ineqs()[8-1]

    find_max_violation()
    # get_ineqs()

    return

if __name__ == '__main__':
    main()
