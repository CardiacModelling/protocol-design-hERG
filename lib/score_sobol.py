from __future__ import print_function
import os
import numpy as np

import pints

from SALib.sample import saltelli
from SALib.analyze import sobol


#
# Setup own score function
#
class MaxSobolScore(pints.ErrorMeasure):
    """
    Self define error measure for square wave protocol optimisation.
    
    This tries to maximise the (mean square) difference between a set of
    input parameter samples function output.
    """
    
    parameters = [
            's1v', 's1t', 's2v', 's2t', 's3v', 's3t'
        ]
    
    def __init__(self, model, problem, max_idx=None, test_func=None, n=100):
        """
        # model: model in this module.
        # problem: a problem dictionary for SALib problem.
        # max_var: the variable index to be maximised its sobol sensitivity.
        # test_func: Pre-test the sobol sequence parameters, take list of
        #            parameters and model as input, return list of True, False
        #            for indicating passed and failed input parameters.
        # n: number of samples; giving n * (n_parameters + 2) simulations.
        """
        super(MaxSobolScore, self).__init__()

        self._model = model
        self._salib_problem = problem
        self._max_idx = max_idx
        self._n_samples = n

        self._dt = 0.5  # ms

        self._n_parameters = len(self.parameters)

        self.set_init_state(None)
        
        self._parameter_samples = saltelli.sample(problem, n,
                calc_second_order=False)
        
        if test_func is not None:
            self._tested_parameters = test_func(
                    self._model,
                    self._parameter_samples)
            print('There are %s out of %s parameters passed the test.' \
                    % (np.sum(self._tested_parameters),
                        len(self._parameter_samples)))
        else:
            self._tested_parameters = None

    def n_parameters(self):
        return self._n_parameters

    def set_max_idx(self, i):
        # Set the variable index to be maximised its sobol sensitivity.
        self._max_idx = i

    def _f(self, trace):
        # Some measure for the time series output to scaler
        # Simply use mean square measure of the time series
        return np.mean(trace ** 2)

    def set_init_state(self, v=None):
        self._set_init_state = v

    def __call__(self, param, get_state=False, debug=False):

        # Time for simulation
        times = np.arange(0, np.sum(param[1::2]), self._dt)
        # Update protocol
        self._model.set_voltage_protocol(param)

        # Run simulations
        f_sims = []
        if get_state:
            states = []

        for i, p in enumerate(self._parameter_samples):
            if self._tested_parameters is not None:
                if self._tested_parameters[i]:
                    pass
                else:
                    f_sims.append(0)
                    if get_state:
                        states.append(None)
                    continue
            if self._set_init_state is not None:
                self._model.set_init_state(self._set_init_state[i])
            # Only use a scaler measure of the simulation output
            # TODO, check if this is sensible...
            f_sims.append(self._f(self._model.simulate(p, times)))
            if get_state:
                states.append(self._model.current_state())
            if not np.any(np.isfinite(f_sims[-1])):
                return np.inf

        if get_state:
            return states

        # Compute Sobol sensitivity
        Si = sobol.analyze(self._salib_problem, np.asarray(f_sims),
                calc_second_order=False)

        # TODO for now, use S1
        to_max = Si['S1'][self._max_idx]  # I think this cannot be negative...

        return -1. * to_max


