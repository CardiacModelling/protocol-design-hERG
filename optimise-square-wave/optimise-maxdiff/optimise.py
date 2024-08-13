#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pints

import model as m
import score_maxdiff as s

savedir = './out'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Get all input variables
import importlib
sys.path.append('./in')  # assume info files are in ./in
try:
    info_id1 = sys.argv[1]
    info1 = importlib.import_module(info_id1)
    info_id2 = sys.argv[2]
    info2 = importlib.import_module(info_id2)
except:
    print('Usage: python %s [str:info_id1] [str:info_id2]' \
            % os.path.basename(__file__))
    sys.exit()

n_repeats = 10

model1 = m.Model(info1.model_file,
        variables=info1.parameters,
        current_readout=info1.current_list,
        set_ion=info1.ions_conc,
        temperature=273.15 + info1.temperature,  # K
        )

model2 = m.Model(info2.model_file,
        variables=info2.parameters,
        current_readout=info2.current_list,
        set_ion=info2.ions_conc,
        temperature=273.15 + info2.temperature,  # K
        )

# protocol parameter: [s1_amp, s1_period, s1_offset, s2_amp, ...], in mV, ms
lower = [0, 750, 0,
        0, 200, 0,
        0, 25, 0]  # Kylie had period = 140, 27, 5 ms
upper = [100, 2e3, 2*np.pi,
        100, 750, 2*np.pi,
        100, 200, 2*np.pi]

class Boundaries(pints.RectangularBoundaries):
    """
    Almost a pints.RectangularBoundaries
    """
    def __init__(self, lower, upper, sum_max=100):
        super(Boundaries, self).__init__(lower, upper)
        self._sum_max = sum_max

    def check(self, parameters):
        """ See :meth:`pints.Boundaries.check()`. """
        if np.any(parameters < self._lower):
            return False
        if np.any(parameters >= self._upper):
            return False
        if np.sum(parameters[::3]) > self._sum_max:
            return False
        print(np.sum(parameters[::3]))
        return True

boundaries = Boundaries(lower, upper, 100)

score = s.MaxDiffScore(model1, model2,
        base_param1=info1.base_param,
        base_param2=info2.base_param,
        total_time=2e3  # ms
        )

x0 = [54, 1400, 0,
    26, 270, 0,
    10, 50, 0]  # square wave version of sine wave!

# Control fitting seed --> OR DONT
# fit_seed = np.random.randint(0, 2**30)
fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)

params, scores = [], []

for _ in range(n_repeats):

    # Try it with x0
    print('Score at x0:', score(x0))
    for _ in range(10):
        assert(score(x0) == score(x0))
    
    
    opt = pints.OptimisationController(
            score,
            x0,
            boundaries=boundaries,
            method=pints.XNES)
    opt.set_max_iterations(None)
    opt.set_parallel(True)

    # Run optimisation
    try:
        with np.errstate(all='ignore'):
            # Tell numpy not to issue warnings
            p, s = opt.run()
            params.append(p)
            scores.append(s)
            print('Found solution:          Sine parameters:' )
            for k, x in enumerate(p):
                print(pints.strfloat(x) + '    ' + \
                        pints.strfloat(x0[k]))
    except ValueError:
        import traceback
        traceback.print_exc()

# Order from best to worst
order = np.argsort(scores)[::]  # (use [::-1] for LL)
scores = np.asarray(scores)[order]
params = np.asarray(params)[order]

# Show results
bestn = min(3, n_repeats)
print('Best %d scores:' % bestn)
for i in range(bestn):
    print(scores[i])
print('Mean & std of logposterior:')
print(np.mean(scores))
print(np.std(scores))
print('Worst logposterior:')
print(scores[-1])

# Extract best
obtained_logposterior0 = scores[0]
obtained_parameters0 = params[0]

#
# Store result
#
with open('%s/opt-diffprt-%s-%s.txt' % (savedir, info1.save_name,
        info2.save_name), 'w') as f:
    f.write('# Amplitude [mV]\tPeriod [ms]\tOffset [1]\n')
    for i in range(len(obtained_parameters0) // 3):
        f.write(pints.strfloat(obtained_parameters0[3 * i]) \
                + '\t' \
                + pints.strfloat(obtained_parameters0[3 * i + 1]) \
                + '\t' \
                + pints.strfloat(obtained_parameters0[3 * i + 2]) \
                + '\n' \
                )

print('Done')
