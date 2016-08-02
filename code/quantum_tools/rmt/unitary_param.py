# ==================
# Parameterization originally found by Spengler, Huber, and Hiesmayr
#   arXiv:1004.5252v2 [quant-ph] 14 Jul 2010
# Heavy Optimizations made by TC Fraser Jul 2016
# ==================

import numpy as np
from scipy import linalg
from quantum_tools.config import *
from quantum_tools.utilities import utils
from functools import reduce
from cmath import exp

def convex_param(θ):
    """
    Given a set of k parameters, find {p_n} such that Σn=1..k+1 p_n = 1
    """

    k = len(θ)
    p = np.zeros(k+1)

    cos2p = np.cos(θ)**2
    sin2p = np.sin(θ)**2

    for n in range(k):
        p[n] = cos2p[n] * np.prod(sin2p[:n])
    p[k] = np.prod(sin2p)

    return p

def is_unitary(U):
    return np.allclose(U.conj().T.dot(U), np.eye(U.shape[0]))

def get_Pl(d):
    """
    The imdepotent projector of P_l = |l><l| where |l> lives in H^d
    """
    Pl = np.zeros((d,d,d))
    for i in range(d):
        Pl[i,i,i] = 1
    return Pl

def expim(M, λ):
    return linalg.expm(1j * M * λ)

def pexpim(M, λ, I):
    return I + M *(exp(1j * λ) - 1)

def sexpim(p_σ, m, n, λ, I):
    return I + (p_σ[m,m] + p_σ[n,n]) * (np.cos(λ) - 1) + (p_σ[m,n] - p_σ[n,m]) * np.sin(λ)

def get_basis(d):
    return np.eye(d)

def get_σ_matrix(d):
    σ = []
    basis = get_basis(d)
    for m in range(d):
        σ_m = []
        for n in range(d):
            partial_σ_mn = 1j * np.outer(basis[n], basis[m])
            σ_mn = partial_σ_mn.conj().T + partial_σ_mn
            σ_m.append(σ_mn)
        σ.append(σ_m)
    return np.array(σ)

def get_partial_σ_matrix(d):
    p_σ = []
    basis = get_basis(d)
    for m in range(d):
        p_σ_m = []
        for n in range(d):
            p_σ_mn = np.outer(basis[m], basis[n])
            p_σ_m.append(p_σ_mn)
        p_σ.append(p_σ_m)
    return np.array(p_σ)

def multi_dot(Mv):
    return reduce(np.dot, Mv)

class UnitaryParam():

    def __init__(self, d):
        self.d = d
        self.basis = get_basis(d)
        self.I = np.eye(d)
        self.Pl = get_Pl(d)
        self.p_σ = get_partial_σ_matrix(d)

    def GP(self, λ):
        return multi_dot(pexpim(self.Pl[l], λ[l,l], self.I) for l in range(self.d))

    def Λ(self, m, n, λ):
        return np.dot(pexpim(self.Pl[n], λ[n,m], self.I), sexpim(self.p_σ, m, n, λ[m,n], self.I))

    def U(self, λ):
        GP = self.GP(λ)
        RRP = multi_dot(multi_dot(self.Λ(m, n, λ) for n in range(m + 1, self.d)) for m in range(self.d-1))
        return np.dot(RRP, GP)


def tests():
    λs = [np.random.normal(0,1,(d, d)) for d in range(1, 5)]
    up = UnitaryParam(3)
    print(up.U(λs[2]))
    print(up.U(λs[2]).conj().T)
    print(np.dot(up.U(λs[2]).conj().T, up.U(λs[2])))
    m, n = 1,2
    p_σ = get_partial_σ_matrix(3)
    print(expim(-1j * p_σ[m,n] + 1j * p_σ[n,m], λ[m,n]))
    print(sexpim(p_σ, m, n, λ[m,n], np.eye(3)))
    print(convex_param(np.array([np.pi/3, np.pi/4, np.pi/6])))

if __name__ == '__main__':
    tests()