from functools import reduce
import numpy as np
from scipy import optimize, linalg
from utils import *
import itertools

# === Configure ===
np.set_printoptions(precision=3, linewidth=120, suppress=True)

# === Constants ===
ineq_coeff = None

# === Memory Locations ===
mem_loc = [2, 2, 2, 2, 7]
ps_A1, ps_A2, ps_B1, ps_B2, ps_rho = gen_memory_slots(mem_loc)

def unpack_param(param):
    p_A1 = param[ps_A1]
    p_A2 = param[ps_A2]
    p_B1 = param[ps_B1]
    p_B2 = param[ps_B2]
    p_rho = param[ps_rho]
    return p_A1, p_A2, p_B1, p_B2, p_rho

def get_context(param):
    p_A1, p_A2, p_B1, p_B2, p_rho = unpack_param(param)
    A1 = get_meas_on_bloch_sphere(*p_A1)
    A2 = get_meas_on_bloch_sphere(*p_A2)
    B1 = get_meas_on_bloch_sphere(*p_B1)
    B2 = get_meas_on_bloch_sphere(*p_B2)
    if (len(p_rho) == 7):
        # === Pure State (switch mem_loc to 7) ===
        psi = get_two_qubit_state(p_rho)
        rho = ket_to_dm(psi)
    elif (len(p_rho) == 16):
        # === Mixed State (switch mem loc to 16) ===
        rho = get_psd_herm(p_rho, 4)
    else:
        raise Exception("Parameterization of state rho needs either 7 or 16 real numbers.")
    return A1, A2, B1, B2, rho

def E(A_x, B_y, rho):
    joint_measure = tensor(A_x, B_y)
    E_AB_xy = np.trace(np.dot(rho, joint_measure))
    # E_AB_xy.imag << 1
    return E_AB_xy.real

def objective(param):
    A1, A2, B1, B2, rho = get_context(param)
    CHSH = E(A1, B1, rho) + E(A1, B2, rho) + E(A2, B1, rho) - E(A2, B2, rho)
    obj = 2 - CHSH
    return obj

def log_results(param):
    print("=== Complete ===")
    print("Results:")
    objective_res = objective(param)
    print("Objective Result", objective_res)
    print("{0} = <A1, B1> + <A1, B2> + <A2, B1> - <A2, B2> <= 2".format(2 - objective_res))
    print("Violation:", 2 - objective_res)
    print("Theo Max Violation:", 2 * np.sqrt(2))

    A1, A2, B1, B2, rho = get_context(param)
    print("A1:")
    print(A1)
    print("A2:")
    print(A2)
    print("A1 * A2:")
    # x = 1/2*(A1 + I2)
    # y = 1/2*(A2 + I2)
    print(np.dot(A1, A2))
    print("B1:")
    print(B1)
    print("B2:")
    print(B2)
    print("B1 * B2:")
    print(np.dot(B1, B2))
    print("rho:")
    print(rho)

def after_iterate(param):
    objective_res = objective(param)
    print("Step result:", objective_res)

def find_max_violation():
    initial_guess = np.random.random((sum(mem_loc))) * -1
    log_results(initial_guess)
    print("=== Minimizing ===")
    res = optimize.minimize(
        fun = objective,
        x0 = initial_guess,
        callback = after_iterate,
    )
    log_results(res.x)

def main():

    # x = get_meas_on_bloch_sphere(np.pi / 4, np.pi / 2)
    # y = get_meas_on_bloch_sphere(np.pi / 4, 0)
    # print(x)
    # print(y)
    # print(np.dot(x,y))

    # return
    a = get_psd_herm(np.arange(16), size=4)
    print(a)
    print(np.dot(a,a))
    print(np.dot(a,a) - a)
    return
    find_max_violation()

if __name__ == '__main__':
    main()