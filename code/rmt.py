import numpy as np
import global_config

def real():
    return np.random.normal()

def complex():
    return real() + 1j * real()

def GL_R(n):
    return np.random.normal(size=((n,n)))

def GL_C(n):
    return np.random.normal(size=((n,n))) \
    + 1j * np.random.normal(size=((n,n)))

def U(n):
    Q, R = np.linalg.qr(GL_C(n))
    r = np.diagonal(R)
    L = np.diag(r/np.abs(r))
    return np.dot(Q, L)

def P_I(n):
    N = range(n)
    S = [np.zeros((n,n)) for _ in N]
    for i in N:
        S_i = np.random.randint(0, n)
        S[S_i][i,i] = 1
    return S

def partial_diag(n):
    empty = np.zeros(n)

def perform_tests():
    # print(random_real())
    # print(complex())
    # print(GL_C(4))
    # print(U(4))
    print(P_I(4))


if __name__ == '__main__':
    perform_tests()