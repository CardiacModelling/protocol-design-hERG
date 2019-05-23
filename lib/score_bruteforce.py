from __future__ import print_function
import numpy as np

import pints


#
# Setup own score function
#
class MaxParamDiffScore(pints.ErrorMeasure):
    """
    Self define error measure for square wave protocol optimisation.
    
    This tries to maximise the (mean square) difference between a set of
    input parameter samples function output.
    """
    
    parameters = [
            's1v', 's1t', 's2v', 's2t', 's3v', 's3t'
        ]
    
    def __init__(self, model, parameter_samples):
        """
        # model: model in this module.
        # parameter_samples: a list of model's parameters that this protocol
        #             optimisation based on.
        """
        super(MaxParamDiffScore, self).__init__()

        self._model = model
        self._parameter_samples = parameter_samples
        self._n_param_samples = len(self._parameter_samples)
        self._n_evaluations = 0.5 * (self._n_param_samples ** 2)

        self._dt = 0.5  # ms

        self._n_parameters = len(self.parameters)
        self._set_voltage = None
        self._set_duration = None

        self.set_voltage(None)
        self.set_duration(None)
        self.set_init_state(None)

    def n_parameters(self):
        return self._n_working_parameters

    def _diff(self, t1, t2):
        return np.mean((t1 - t2)**2)

    def set_voltage(self, v=None):
        self._set_voltage = v
        if v is None and self._set_duration is None:
            self._n_working_parameters = self._n_parameters
        elif v is not None and self._set_duration is None:
            self._n_working_parameters = int(self._n_parameters / 2)
        else:
            raise ValueError('Both voltage and duration are fixed.')

    def set_duration(self, v=None):
        self._set_duration = v
        if v is None and self._set_voltage is None:
            self._n_working_parameters = self._n_parameters
        elif v is not None and self._set_voltage is None:
            self._n_working_parameters = int(self._n_parameters / 2)
        else:
            raise ValueError('Both voltage and duration are fixed.')

    def set_init_state(self, v=None):
        self._set_init_state = v

    def __call__(self, param, get_state=False, debug=False):

        if self._set_voltage is not None and self._set_duration is not None:
            raise ValueError('Both voltage and duration are fixed.')

        if self._set_voltage is not None:
            param_duration = np.copy(param)
            param = np.zeros(6)
            param[::2] = np.copy(self._set_voltage)
            param[1::2] = param_duration

        if self._set_duration is not None:
            param_voltage = np.copy(param)
            param = np.zeros(6)
            param[1::2] = np.copy(self._set_duration)
            param[::2] = param_voltage

        # Time for simulation
        times = np.arange(0, np.sum(param[1::2]), self._dt)
        # Update protocol
        self._model.set_voltage_protocol(param)

        # Run simulations
        sims = []
        if get_state:
            states = []

        for i, p in enumerate(self._parameter_samples):
            if self._set_init_state is not None:
                self._model.set_init_state(self._set_init_state[i])
            sims.append(self._model.simulate(p, times))
            if get_state:
                states.append(self._model.current_state())
            if not np.any(np.isfinite(sims[-1])):
                return np.inf

        if get_state:
            return states

        # Compute differences
        total_diff = 0
        for i1 in range(self._n_param_samples):
            t1 = sims[i1]
            for i2 in range(i1 + 1, self._n_param_samples):
                t2 = sims[i2]
                total_diff += self._diff(t1, t2)

        return -1. * total_diff / self._n_evaluations


