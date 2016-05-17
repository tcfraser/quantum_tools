import time

def timing(f):
    def wrap(*args):
        t_i = time.time()
        ret = f(*args)
        t_f = time.time()
        print('[timing_profiler] Function "{func}" took {t:.3f} seconds.'.format(func = f.__name__, t = (t_f-t_i)))
        return ret
    return wrap