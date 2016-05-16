from functools import reduce
import numpy as np
from scipy import optimize, linalg
from ineqs_loader import get_ineqs
from utils import *
import itertools
# from sklearn.preprocessing import normalize
# from qutip import Qobj, tensor, ket2dm, basis, sigmax, sigmay, sigmaz, expect, qeye, qstate

# === Configure ===
np.set_printoptions(precision=3, linewidth=120, suppress=True)

# === Constants ===
ineq_coeff = None

def get_projective_meas(weight, param):
    # weight = min(max(weight, 0), 1)
    assert (len(param) == 7), "Projective state needs 7 parameters."
    # assert (0 <= weight <= 1), "Weight {w} is not normalized.".format(w=weight)
    state = get_two_qubit_state(param)
    M_y = norm_real_parameter(weight) * ket_to_dm(state)
    M_n = I4 - M_y
    dM = M_y - M_n
    return dM

def get_meas(weight, param):
    M_y = norm_real_parameter(weight) * get_mixed_measurement(param)
    M_n = I4 - M_y
    dM = M_y - M_n
    return dM

def get_mixed_measurement(param):
    return get_psd_herm(param, 4, unitary_trace=True)

def get_density_matrix(param):
    return get_psd_herm(param, 4, unitary_trace=True)

def get_perumation():
    perm = np.zeros((64,64), dtype=complex)
    buffer = np.empty((64,64), dtype=complex)
    for a in list(itertools.product(*[[0,1]]*6)):
        ket = tensor(*(qbs[a[i]] for i in (0,1,2,3,4,5)))
        bra = tensor(*(qbs[a[i]] for i in (1,2,3,4,5,0)))
        np.outer(ket, bra, buffer)
        perm += buffer
    return perm

# === Memory Locations ===
mem_loc = [1,16,1,16,1,16,16,16,16]
p_wA, p_mA, p_wB, p_mB, p_wC, p_mC, p_sX, p_sY, p_sZ = gen_memory_slots(mem_loc)

def unpack_param(param):
    # print(p_wA, p_mA, p_wB, p_mB, p_wC, p_mC, p_sX, p_sY, p_sZ)
    # print(param)
    wA = param[p_wA][0]
    mA = param[p_mA]
    wB = param[p_wB][0]
    mB = param[p_mB]
    wC = param[p_wC][0]
    mC = param[p_mC]
    sX = param[p_sX]
    sY = param[p_sY]
    sZ = param[p_sZ]
    return wA, mA, wB, mB, wC, mC, sX, sY, sZ

def get_context(param):
    wA, mA, wB, mB, wC, mC, sX, sY, sZ = unpack_param(param)
    dA = get_meas(wA, mA)
    dB = get_meas(wB, mB)
    dC = get_meas(wC, mC)
    sX = get_density_matrix(sX)
    sY = get_density_matrix(sY)
    sZ = get_density_matrix(sZ)
    return dA, dB, dC, sX, sY, sZ

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

def explode_correlator(c):
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

# def bounded_weight(param, wI4):
#     wA, mA, wB, mB, wC, mC, sX, sY, sZ = unpack_param(param)
#     # print(wI4)
#     w = [wA, wB, wC][wI4]
#     # print('weight', w)
#     return 1/2 - abs(w - 1/2)

def objective(param):
    dA, dB, dC, sX, sY, sZ = get_context(param)
    perm = get_perumation()

    XxYxZ = tensor(sX, sY, sZ)
    state = reduce(np.dot, (perm.T, XxYxZ, perm))
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
    c = gen_correlator(diffM, state)

    target_value = np.dot(ineq_coeff, explode_correlator(c))
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
    A = get_mixed_measurement(mA)
    B = get_mixed_measurement(mB)
    C = get_mixed_measurement(mC)
    X = get_density_matrix(sX)
    Y = get_density_matrix(sY)
    Z = get_density_matrix(sZ)
    print("Measure State A (w={0}):".format(norm_real_parameter(wA)))
    print(A)
    print("Measure State B (w={0}):".format(norm_real_parameter(wB)))
    print(B)
    print("Measure State C (w={0}):".format(norm_real_parameter(wC)))
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
    initial_guess = np.random.random((sum(mem_loc))) * -1
    # print(initial_guess)
    # log_results(initial_guess)
    # return
    print("=== Minimizing ===")
    res = optimize.minimize(
        fun = objective,
        x0 = initial_guess,
        callback = after_iterate,
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
