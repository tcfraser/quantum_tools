import numpy as np
from scipy import linalg

# === Utils ===
def gen_memory_slots(mem_loc):
    i = 0
    slots = []
    for m_size in mem_loc:
        slots.append(np.arange(i, i+m_size))
        i += m_size
    return slots

def normalize(a):
    a /= linalg.norm(a)

def norm_real_parameter(x):
    return np.cos(x)**2