from ineqs_loader import get_ineqs
import itertools
from triangle_violation import TriangleUtils
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Computer modern fonts
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# === Configure ===
np.set_printoptions(precision=3, linewidth=120, suppress=True)

def get_exploded_correlator(A,B,C):
    correl = {}
    correl['_'] = 1
    correl['A'] = A
    correl['B'] = B
    correl['C'] = C
    correl['AB'] = A*B
    correl['BC'] = B*C
    correl['AC'] = A*C
    correl['ABC'] = A*B*C
    exploded_correlator = TriangleUtils.explode_correlator(correl)
    return exploded_correlator

def main():
    ineqs = get_ineqs()
    num_ineqs = len(ineqs)
    target = np.empty((num_ineqs, 8))
    ineq_index = 0
    models = list(itertools.product(*[[-1,1]]*3))
    print(models)
    for ineq_index in range(num_ineqs):
        for det_model, model_data in enumerate(models):
            exploded_correlator = get_exploded_correlator(*model_data)
            # print(exploded_correlator)
            ineq_value = np.dot(ineqs[ineq_index], exploded_correlator)
            target[ineq_index, det_model] = ineq_value
    print(target)

    for ineq_values in target:
        if not np.any(ineq_values == 0):
            print("Deterministic model failed.")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(target, interpolation='nearest')
    fig.colorbar(cax)

    plt.yticks(np.arange(0, num_ineqs), np.arange(1, num_ineqs+1))
    plt.xticks(np.arange(8), (str(model) for model in models))
    plt.xticks(rotation = 90)

    plt.show()


    return target

if __name__ == "__main__":
    # a = get_exploded_correlator(1,-1,-1)
    # print(a)
    # print(get_ineqs()[35-1])
    # print(np.dot(get_ineqs()[35-1], a))
    # pass
    main()
