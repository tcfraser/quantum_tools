from multiprocessing import Process, Queue, Pool
import multiprocessing
import math
from pprint import pprint
# from timing_profiler import timing

class JobContext():

    def __init__(self, target_func, target_args, num_cores=-1, processes_per_core=1, log_worker=False):
        self.num_evals = len(target_args)
        self.target_args = target_args
        self.target_func = target_func
        self.num_evals = len(target_args)
        if num_cores <= 0:
            self.num_cores = multiprocessing.cpu_count() - 1
        else:
            self.num_cores = min(num_cores, multiprocessing.cpu_count())
        print("JobContext using {0} cores.".format(self.num_cores))
        self.processes_per_core = processes_per_core
        self.log_worker = log_worker
        self.target_results = []

    def _store_result(self, result):
        print("Sub-Job Finished.")
        self.target_results.append(result)

    def evaluate(self):
        out_queue = Queue()
        num_processes = self.num_cores * self.processes_per_core
        pool = Pool(processes = num_processes)
        for i in range(self.num_evals):
            pool.apply_async(self.target_func, args=self.target_args[i], callback=self._store_result)
        pool.close()
        pool.join()

def f(a,b,c):
    return a - b * c

def main():
    jc = JobContext(f, [[1,2,3], [2,3,4], [3,4,5], [3, 4, 5], [3, 4, 11]])
    jc.evaluate()
    for result in jc.target_results:
        print(result)

if __name__ == '__main__':
    main()