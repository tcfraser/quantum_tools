import numpy as np
from ..config import *

def get_ineqs():
    # print("Loading inequalities...")
    file = open(SOURCE_DIR + os.path.join(os.sep, 'examples', 'ineqs.csv'), 'rb')
    ineqs = np.loadtxt(file, delimiter=',')
    file.close()
    # print("Loaded inequalities with shape: {0}".format(ineqs.shape))
    # print(ineqs)
    return ineqs

def get_ineq(i):
    return get_ineqs()[i]
