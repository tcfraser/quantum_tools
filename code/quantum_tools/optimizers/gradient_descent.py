from scipy import optimize
import multiprocessing as mp

def sym_gradient(f, x, dx):
    return (gradient(f,x,dx) + gradient(f,x,-dx)) / 2

def get_f_xdx(f, x, dx):
    f_xdx = np.zeros(f.param_size)
    for i in range(f.param_size):
        x[i] += dx
        f_xdx[i] = f(x)
        x[i] -= dx
    return f_xdx

def gradient(f, x, dx):
    return (get_f_xdx(f, x, dx) - f(x))/dx

def gradient_descent(f, x0, dx, gamma, n_max, adaptive=False):
    n = 0
    x = np.asarray(x0, dtype='float64')
    assert(f.param_size == len(x))
    h_f = np.zeros(n_max)
    h_x = np.zeros((n_max, f.param_size))
    while n < n_max:


        f_x = f(x)
        print(n, f_x, x)
        # Update History
        h_f[n] = f_x
        h_x[n, :] = x
        f_xdx = get_f_xdx(f, x, dx)
        df_x = (f_xdx - f_x)/dx

#         df_x[df_x > 0] = 0

        n += 1
        if not adaptive:
            x = x - gamma * df_x
        else:
            while True:
                x_new = x - gamma * df_x
                f_x_new = f(x_new)

                delta_f = f_x - f_x_new
                if delta_f >= -0.001: # Previous value greater
                    gamma *= math.exp(delta_f) # Get Larger
                    break
                else:
                    if gamma < 1e-5:
                        break
                    gamma *= math.exp(delta_f) # Get Smaller
            x = x_new
    return h_f, h_x
#     print(mx)
#     return x

# PROFILE_MIXIN(gradient_descent, f, x0, 0.01, 0.001)
#
# print("Done")
# TESTING GRADIENT DESCENT
h_f, h_x = gradient_descent(Fxy, [1, 1], 0.01, 0.1, 100, adaptive=True)

import math
def Fxy(x, y):
    return math.sin(x**2 / 2 - y**2 / 4 + 3) * math.cos(2*x + 1 - math.e**y)

Fxy = Caller(Fxy, 2)

class Caller():

    def __init__(self, f, size):
        self._f = f
        self.param_size = size

    def __call__(self, param):
        return self._f(*param)
