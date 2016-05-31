from __future__ import print_function, division
import numpy as np

i = 1j
mach_tol = 1e-12
mach_eps = 1e-20
I2 = np.eye(2)
I4 = np.eye(4)
qb0 = np.array([[1], [0]], dtype=complex) # Spin down (ground state)
qb1 = np.array([[0], [1]], dtype=complex) # Spin up (excited state)
qbs = np.array([qb0, qb1])
qb00 = np.kron(qb0, qb0)
qb01 = np.kron(qb0, qb1)
qb10 = np.kron(qb1, qb0)
qb11 = np.kron(qb1, qb1)
sigx = np.array([[0,1],[1,0]])
sigy = np.array([[0,-i],[i,0]])
sigz = np.array([[1,0],[0,-1]])
tqbs = np.array([qb00, qb01, qb10, qb11])