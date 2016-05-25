from multiprocessing import Process, Queue
import multiprocessing
import math
from pprint import pprint
# from timing_profiler import timing

class JobContext():

    def __init__(self, target_func, target_args, num_cores=-1, processes_per_core=1, log_worker=False):
        self.num_evals = len(target_args)
        self.target_args = target_args
        self.target_func = target_func
        if num_cores <= 0:
            self.num_cores = multiprocessing.cpu_count() - 1
        else:
            self.num_cores = min(num_cores, multiprocessing.cpu_count())
        self.processes_per_core = processes_per_core
        self.log_worker = log_worker
        self.target_results = []

    def worker(self, out_queue, sub_target_args):
        for args in sub_target_args:
            eval_result = self.target_func(*args)
            # Maybe custom pickler here
            out_queue.put(eval_result)

    def evaluate(self):
        out_queue = Queue()
        num_processes = self.num_cores * self.processes_per_core
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
                print("Sub-Job {i} Finished.".format(i=i))
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