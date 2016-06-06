import numpy as np

def real():
    return np.random.normal()

def complex():
    return real() + 1j * real()

def GL_R(n):
    return np.random.normal(size=((n,n)))

def GL_C(n):
    return np.random.normal(size=((n,n))) \
    + 1j * np.random.normal(size=((n,n)))

def GL_knit_QR(GL_n):
    Q, R = np.linalg.qr(GL_n)
    r = np.diagonal(R)
    L = np.diag(r/np.abs(r))
    return np.dot(Q, L)

def U(n):
    return GL_knit_QR(GL_C(n))

def P_I(size, count):
    S = [np.zeros((size,size)) for _ in range(count)]
    for i in range(size):
        S[np.random.randint(0, count)][i,i] = 1
    return S

# def P_I(n):
#     N = range(n)
#     S = [np.zeros((n,n)) for _ in N]
#     for i in N:
#         for j in N:
#             if (real() > 0):
#                 S[j][i,i] = 1
#     return S

def perform_tests():
    # print(random_real())
    # print(complex())
    # print(GL_C(4))
    # print(U(4))
    print(P_I(4))


if __name__ == '__main__':
    perform_tests()