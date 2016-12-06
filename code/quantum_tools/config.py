import numpy
import os
from functools import reduce
import sys

# === Configure ===
numpy.set_printoptions(precision=5, linewidth=120, suppress=True)
__directory = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
__root_dir = os.path.dirname(__directory)
__path_to_output = os.path.join(os.sep, 'examples', 'outputs')

NOTEBOOK_FILES_DIR = __root_dir
ROOT_DIR = __root_dir
SOURCE_DIR = __directory
OUTPUT_DIR = __directory + __path_to_output + os.sep # Why python is this so hard?
BULK_DIR = os.path.join(os.path.dirname(__directory), 'bulk')

ASSERT_MODE = 0

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
        # ps.strip_dirs()
        ps.sort_stats('cumulative')
        # ps.print_stats()
        ps.print_stats(.2)
        if error is not None:
            raise error

# print(OUTPUT_DIR)