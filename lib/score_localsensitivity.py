from __future__ import print_function
import os
import sys
sys.path.append('../lib')
import numpy as np

import pints
import myokit
import myokit.pacing as pacing


#
# Setup own score function
#
class NaiveAllS1Score(pints.ErrorMeasure):
    """
    Self define error measure for 3-step protocol optimisation.
    
    Note that this is not normalised to 1, so it is $J_i / \sum_{k!=i} J_k$,
    where $J_i = \int_{t over step 3} |dI/dp_i * p_i| dt$.
    """
    
    parameters = [
            's1v', 's1t', 's2v', 's2t', 's3v', 's3t'
        ]
    
    def __init__(self, model, max_idx, base_param, var_list,
                 sensitivity=0.025):
        """
        # model: model in this module.
        # max_idx: idx of the current list for which to be maximised; this
        #          should match `base_param`.
        # base_param: model's (local) parameters that this protocol
        #             optimisation based on; this should match `var_list`.
        # var_list: {'variable': 'current', ...}.
        # sensitivity = naive approx. to sensitivity.
        """
        super(NaiveAllS1Score, self).__init__()

        self._model = model
        self._max_idx = max_idx
        self._base_param = base_param
        self._var_list = var_list
        self._variables = model.parameter()
        self._max_var = self._variables[self._max_idx]
        self._sensitivity = sensitivity
        self._dt = 0.5  # ms

    def n_parameters(self):
        return len(self.parameters)

    def _naiveS1(self, times, idx_i, idx_f, debug=False):
        total = 0
        tomax = None  # make sure it won't do something silly if tomax not def
        for i, var in enumerate(self._variables):
            cur = self._var_list[var]

            if debug:
                print(var, cur)
            # p_i = p_i + sensitivity * p_i
            x = np.copy(self._base_param)
            x[i] = x[i] * (1 + self._sensitivity)
            d1 = self._model.simulate(x, times, [cur])
            # p_i = p_i - sensitivity * p_i
            x = np.copy(self._base_param)
            x[i] = x[i] * (1 - self._sensitivity)
            d_1 = self._model.simulate(x, times, [cur])
            
            if d1 == float('inf') or d_1 == float('inf'):
                if debug:
                    print(d1, d_1)
                return float('inf')

            dIdvar = (d1[cur][idx_i:idx_f] - d_1[cur][idx_i:idx_f]) \
                    / (2 * self._sensitivity)

            if var != self._max_var:
                total += np.sum(np.abs(dIdvar))
                if debug:
                    print(np.sum(np.abs(dIdvar)))
            else:
                tomax = np.sum(np.abs(dIdvar))
                if debug:
                    print(-np.sum(np.abs(dIdvar)))

        return total, tomax

    def __call__(self, param, debug=False):
        # Time for simulation
        times = np.arange(0, np.sum(param[1::2]), self._dt)
        pre_time = np.sum(param[1:-1:2])
        idx_i = int(pre_time / self._dt)
        idx_f = -1
        # Update protocol
        self._model.set_voltage_protocol(param)

        total, tomax = self._naiveS1(times, idx_i, idx_f, debug)

        return - tomax / total  # or total / tomax


class NaiveAllS1CurrentScore(NaiveAllS1Score):
    """
    Self define error measure for 3-step protocol optimisation.
    
    Note that this is not normalised to 1, so it is $J_i / \sum_{k!=i} J_k$,
    where $J_i = \int_{t over step 3} |dI/dp_i * p_i| dt$.
    """

    def __init__(self, model, max_idx, base_param, var_list,
                 sensitivity=0.025):
        """
        # model: model in this module.
        # max_idx: idx of the parameter for which to be maximised; this
        #          should match `base_param`.
        # base_param: model's (local) parameters that this protocol
        #             optimisation based on; this should match `var_list`.
        # var_list: {'variable': 'current', ...}.
        # sensitivity = naive approx. to sensitivity.
        """
        super(NaiveAllS1CurrentScore, self).__init__(
                model, max_idx, base_param, var_list, sensitivity)

    def __call__(self, param, debug=False):
        # Time for simulation
        times = np.arange(0, np.sum(param[1::2]), self._dt)
        pre_time = np.sum(param[1:-1:2])
        idx_i = int(pre_time / self._dt)
        idx_f = -1
        # Update protocol
        self._model.set_voltage_protocol(param)
        
        # Part 1: parameter sensitivity
        total, tomax = self._naiveS1(times, idx_i, idx_f, debug)

        # Part 2: make current size bigger

        ## p_i = p_i
        x = np.copy(self._base_param)
        max_cur = self._var_list[self._max_var]
        d0 = self._model.simulate(x, times, [max_cur])

        mean_o = 0.3  # open prob that roughly gives a good current
        mean_c = np.mean(np.abs(d0[max_cur][idx_i:idx_f])) / \
                (mean_o * self._base_param[0] * 100) # ~100 mV for (V - EK)
        w_mean_c = 1.5  # weighting of mean_c

        return - tomax / total - w_mean_c * mean_c  # or total / tomax


class NormalisedNaiveS1CurrentScore(NaiveAllS1Score):
    """
    Self define error measure for 3-step protocol optimisation.
    
    This is normalised to 2, so it the sum of $J_i / \sum_{k} J_k$, where
    $J_i = \int_{t over step 3} |dI/dp_i * p_i| dt$, and 
    $\int_{t over step 3} |I_j| dt / \sum{j} \int_{t over step 3} |I_j| dt$.
    
    A caveat is that the second criterion does not guarantee an overall big
    current.
    """
    
    def __init__(self, model, max_idx, base_param, var_list,
                 sensitivity=0.025):
        """
        # model: model in this module.
        # max_idx: idx of the parameter for which to be maximised; this
        #          should match `base_param`.
        # base_param: model's (local) parameters that this protocol
        #             optimisation based on; this should match `var_list`.
        # var_list: {'variable': 'current', ...}.
        # sensitivity = naive approx. to sensitivity.
        """
        super(NormalisedNaiveS1CurrentScore, self).__init__(
                model, max_idx, base_param, var_list, sensitivity)

    def __call__(self, param, debug=False):
        # Time for simulation
        times = np.arange(0, np.sum(param[1::2]), self._dt)
        pre_time = np.sum(param[1:-1:2])
        idx_i = int(pre_time / self._dt)
        idx_f = -1
        # Update protocol
        self._model.set_voltage_protocol(param)
        
        # p_i = p_i
        x = np.copy(self._base_param)
        d0_all = self._model.simulate(x, times, self._current_list)

        # Part 1
        total = 0
        tomax = None  # make sure it won't do something silly if tomax not def
        for i, var in enumerate(self._variables):
            cur = self._var_list[var]
            
            if debug:
                print(var, self._var_list[var])
            # p_i = p_i + sensitivity * p_i
            x = np.copy(self._base_param)
            x[i] = x[i] * (1 + self._sensitivity)
            d1 = self._model.simulate(x, times, [cur])
            # p_i = p_i - sensitivity * p_i
            x = np.copy(self._base_param)
            x[i] = x[i] * (1 - self._sensitivity)
            d_1 = self._model.simulate(x, times, [cur])
            
            if d1 == float('inf') or d_1 == float('inf'):
                if debug:
                    print(d1, d_1)
                return float('inf')

            dIdvar = (d1[cur][idx_i:idx_f] - d_1[cur][idx_i:idx_f]) \
                    / (2 * self._sensitivity)

            if var != self._max_var:
                total += np.sum(np.abs(dIdvar))
                if debug:
                    print(np.sum(np.abs(dIdvar)))
            else:
                tomax = np.sum(np.abs(dIdvar))
                total += tomax
                if debug:
                    print(-np.sum(np.abs(dIdvar)))
                
        # Part 2
        total_c_not_i = []
        tomax_c = None
        for i in self._current_list:
            if i != self._var_list[self._max_var]:
                total_c_not_i.append(np.abs(d0[i][idx_i:idx_f]))
            else:
                tomax_c = np.abs(d0[i][idx_i:idx_f])
                total_c_not_i.append(tomax_c)
                print(i)
        tomax_c = np.sum(tomax_c)
        total_c_not_i = np.sum(total_c_not_i)
        
        return -1 * (tomax / total + tomax_c / total_c_not_i)


