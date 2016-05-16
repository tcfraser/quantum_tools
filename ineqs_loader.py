import numpy as np

def get_ineqs():
    print("Loading inequalities...")
    file = open('ineqs.csv', 'rb')
    ineqs = np.loadtxt(file, delimiter=',')
    # print("Loaded inequalities with shape: {0}".format(ineqs.shape))
    print(ineqs)
    return ineqs