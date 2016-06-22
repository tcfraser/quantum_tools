import numpy
import os
from functools import reduce
import sys

# === Configure ===
numpy.set_printoptions(precision=3, linewidth=120, suppress=True)
__directory = os.path.dirname(os.path.abspath(__file__))
__path_to_output = os.path.join(os.sep, 'examples', 'outputs')

SOURCE_DIR = __directory
OUTPUT_DIR = __directory + __path_to_output + os.sep # Why python is this so hard?

def STOP():
    sys.exit(0)

def PROFILE_MIXIN(func, *args):
    import cProfile, pstats
    pr = cProfile.Profile()
    pr.enable()
    error = None
    try:
        func(*args)
    except Exception as e:
        error = e
    finally:
        pr.disable()
        # ps = pstats.Stats(pr).sort_stats('tottime')
        ps = pstats.Stats(pr)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats()
        # ps.print_stats(.1)
        if error is not None:
            raise error

# print(OUTPUT_DIR)