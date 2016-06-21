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

# print(OUTPUT_DIR)