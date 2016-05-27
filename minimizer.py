from __future__ import print_function, division
import numpy as np
from scipy import optimize, linalg
from utils import Utils
from pprint import pprint
import itertools
from job_queuer_async import JobContext
from timing_profiler import timing
import global_config
from global_config import *

class Minimizer():

    def __init__(self, mem_loc):
        self.__solved__ = False
        self.best_objective_result = np.inf
        self.best_objective_result_param = None
        self.string_logs = []
        self.target_value = 0
        self.local_log = False
        self.stepsize = 1e4
        self.tolerance = 1e-2
        self._num_evals = 0
        self.max_evals = 1
        self.mem_loc = mem_loc
        self.mem_slots = Utils.gen_memory_slots(self.mem_loc)
        self.mem_size = sum(self.mem_loc)
        self.log('Initialized')

    def log(self, log_item, *args):
        if isinstance(log_item, str) and args is not None:
            log_item = log_item.format(*args)
        if self.local_log:
            pprint(log_item)
        self.string_logs.append([log_item])

    def _basin_hopping_callback(self, param, objective_res, accept):
        self.log("BH Step result: {0}", objective_res)
        self.log("Local Minimum Accepted?: {0}", accept)
        self._num_evals += 1
        if (objective_res < self.best_objective_result):
            self.best_objective_result = objective_res
            self.best_objective_result_param = param
            self.log("New Best Objective Result: {0}", objective_res)
        reached_target = (objective_res <= self.target_value)
        hit_max_iters = (self._num_evals >= self.max_evals)
        should_terminate = reached_target or hit_max_iters
        return should_terminate

    def _minimize_callback(self, param):
        objective_res = self.objective(param)
        self.log("Minimize Step result: {0}", objective_res)

    def initial_guess(self,):
        """
        @Overwrite
        """
        return np.random.normal(scale=1.0, size=self.mem_size)

    def get_context(self, param):
        """
        @Overwrite
        """
        raise NotImplemented()

    def get_prob_distrobution(self, context):
        """
        @Overwrite
        """
        raise NotImplemented()

    def objective(self, param):
        """
        @Overwrite
        """
        raise NotImplemented()

    def minimize(self):
        self._basin_hopping()

    def _basin_hopping(self):
        if self.__solved__:
            raise Exception("Already Solved")

        res = optimize.basinhopping(
            func = self.objective,
            x0 = self.initial_guess(),
            callback = self._basin_hopping_callback,
            # T = 1e-2,
            stepsize = self.stepsize,
            minimizer_kwargs = {
                'method': 'L-BFGS-B',
                'tol': self.tolerance,
                'callback': self._minimize_callback
            },
        )
        self.__solved__ = True
        self.log("Solved")
        self.best_context = self.get_context(self.best_objective_result_param)
        self.best_pd = self.get_prob_distrobution(self.best_context)