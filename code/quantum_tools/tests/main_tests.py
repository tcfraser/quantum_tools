from ..config import *
from ..contexts.measurement import Measurement
from ..utilities.timing_profiler import timing
from ..utilities import utils
import numpy as np

def save_file_to_output():
    with open(OUTPUT_DIR + 'test.txt', 'w+') as f:
        f.write('This is a successful test.')
    print("Wrote to file.")

@timing
def print_tests():
    print(utils.param_GL_C(np.random.normal(0,1,32)))
    return
    m = Measurement.Strats.Deterministic.deterministic('A', 4)
    print(m)
    m = Measurement.Strats.Random.seesaw('A', 4)
    print(m)
    m = Measurement.Strats.Random.pvms('A', 4)
    print(m)
    print(Measurement)

if __name__ == '__main__':
    print_tests()
    # save_file_to_output()