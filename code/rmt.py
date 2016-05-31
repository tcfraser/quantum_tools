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

def partial_diag(n):
    empty = np.zeros(n)

def perform_tests():
    # print(random_real())
    # print(complex())
    # print(GL_C(4))
    print(U(4))


if __name__ == '__main__':
    perform_tests()