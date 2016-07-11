from __future__ import print_function, division
import numpy as np
import itertools
from pprint import pprint
from scipy import optimize, linalg
from ..utilities import utils
from ..config import *

class Minimizer():

    def __init__(self, mem_loc):
        self.__solved__ = False
        self.best_objective_result = np.inf
        self.best_objective_result_param = None
        self.string_logs = []
        self.target_value = -np.inf
        self.local_log = False
        self.stepsize = 10
        self.tolerance = 1e-3
        self._num_evals = 0
        self.max_evals = 1
        self.mem_loc = mem_loc
        self.mem_slots = utils.gen_memory_slots(self.mem_loc)
        self.mem_size = sum(self.mem_loc)
        self.log('Initialized')

    def log(self, *args):
        str_args = ', '.join(str(arg) for arg in args)
        if self.local_log:
            pprint(str_args)
        self.string_logs.append(str_args)

    def _basin_hopping_callback(self, param, objective_res, accept):
        self.log("BH Step result: {0}".format(objective_res))
        self.log("Local Minimum Accepted?: {0}".format(accept))
        self._num_evals += 1
        if (objective_res < self.best_objective_result):
            self.best_objective_result = objective_res
            self.best_objective_result_param = param
            self.log("New Best Objective Result: {0}".format(self.best_objective_result))
        reached_target = (objective_res <= self.target_value)
        hit_max_iters = (self._num_evals >= self.max_evals - 1) # -1 here because basinhopping has initialization step
        should_terminate = reached_target or hit_max_iters
        if reached_target:
            self.log("Reached target.")
        if hit_max_iters:
            self.log("Hit max iterations.")
        return should_terminate

    def _minimize_callback(self, param):
        objective_res = self.objective(param)
        self.log("Minimize Step result: {0}".format(objective_res))
        if (objective_res < self.best_objective_result):
            self.best_objective_result = objective_res
            self.best_objective_result_param = param
            self.log("New Best Objective Result: {0}".format(self.best_objective_result))

    def initial_guess(self):
        """
        @Overwrite
        """
        return np.random.normal(scale=1.0, size=self.mem_size)

    def get_context(self, param):
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
        # self.best_pd = self.get_prob_distribution(self.best_context)

    def save_results_to_file(self, file_name):
        if not self.__solved__:
            raise Exception("Not Solved Yet")

        with open(file_name, 'w+') as file_:
            file_.write("Best Objective Result:")
            file_.write("\n")
            file_.write(str(self.best_objective_result_param))
            file_.write("\n")
            # file_.write("Best Probability Distribution:")
            # file_.write("\n")
            file_.write("Best Context:")
            file_.write("\n")
            file_.write(str(self.best_context))
            file_.write("\n")
            file_.write("Logs:")
            file_.write("\n")
            for log in self.string_logs:
                file_.write(str(log))
                file_.write("\n")
