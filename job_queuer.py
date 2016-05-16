from multiprocessing import Process, Queue
import multiprocessing
import math
from pprint import pprint
# from timing_profiler import timing

N_CORES = multiprocessing.cpu_count() - 1
# N_CORES = 1
PROCESSES_PER_CORE = 1

class JobContext():

    def __init__(self, target_func, target_args, log_worker=False):
        self.num_evals = len(target_args)
        self.target_args = target_args
        self.target_func = target_func
        self.log_worker = log_worker
        self.target_results = []

    def worker(self, out_queue, sub_target_args):
        for args in sub_target_args:
            eval_result = self.target_func(*args)
            # Maybe custom pickler here
            out_queue.put(eval_result)

    def evaluate(self):
        out_queue = Queue()
        num_processes = N_CORES * PROCESSES_PER_CORE
        evals_per_process = math.ceil(self.num_evals / num_processes)
        processes = []

        i = 0
        while i < self.num_evals:
            remaining_evals = self.num_evals - i
            eval_batch_size = min(remaining_evals, evals_per_process)
            process = Process(
                target=self.worker,
                args=(out_queue, self.target_args[i:i+eval_batch_size]))
            processes.append(process)
            process.start()
            i += eval_batch_size

        for i in range(self.num_evals):
            serialized_eval_result = out_queue.get()
            if self.log_worker:
                print("Worker {i} Finished.".format(i=i))
                # pprint(serialized_eval_result)
            self.target_results.append(serialized_eval_result)

        # for process in processes:
        #     process.daemon = True
        for process in processes:
            process.join()

def f(a,b,c):
    return a - b * c

def main():
    jc = JobContext(f, [[1,2,3], [2,3,4], [3,4,5], [3, 4, 5], [3, 4, 11]])
    jc.evaluate()
    for result in jc.target_results:
        print(result)

if __name__ == '__main__':
    main()