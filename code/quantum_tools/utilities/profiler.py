import time
import os
import psutil

def max_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def profile(f):
    def wrap(*args):
        t_i = time.time()
        ret = f(*args)
        t_f = time.time()
        mem = max_memory()
        print('[profiler] Function "{func}" took {t:.3f} seconds. Used {mem} memory.'.format(func = f.__name__, t = (t_f-t_i), mem = sizeof_fmt(mem)))
        return ret
    return wrap