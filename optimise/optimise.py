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

savedir = './out'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Get all input variables
import importlib
sys.path.append('./in')  # assume info files are in ./in
try:
    info_id = sys.argv[1]
    info = importlib.import_module(info_id)
except:
    print('Usage: python %s [str:info_id]' % os.path.basename(__file__))
    sys.exit()

n_repeats = 1

model = m.Model(info.model_file,
        variables=info.parameters,
        current_readout=info.current_list,
        set_ion=info.ions_conc,
        temperature=273.15 + info.temperature,  # K
        )

# protocol parameter: [step_1_voltage, step_1_duration, step_2..., step3...]
lower = [-120, 50,
        -120, 50,
        -120, 50,]
upper = [60, 1e3,
        60, 1e3,
        60, 1e3,]
boundaries = pints.RectangularBoundaries(lower, upper)

sensitivity = 0.025
params = []
scores = []
for i_var, var in enumerate(info.parameters):
    # Control fitting seed --> OR DONT
    # fit_seed = np.random.randint(0, 2**30)
    fit_seed = 542811797
    print('Fit seed: ', fit_seed)
    np.random.seed(fit_seed)

    print('Optimising parameter %s in %s' % (var, info.var_list[var]))
    best_s = np.inf
    best_p = None
    for _ in range(n_repeats):
        score = m.NaiveAllS1CurrentScore(model,
                max_idx=i_var,
                base_param=info.base_param,
                var_list=info.var_list,
                sensitivity=sensitivity)

        x0 = [-80, 5e2,
             -80, 5e2,
             -80, 5e2,]

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
            with np.errstate(all='ignore'): # Tell numpy not to issue warnings
                p, s = opt.run()
                if s < best_s:
                    best_s = s
                    best_p = p
        except ValueError:
            import traceback
            traceback.print_exc()

        print('var: ', var)
        print('found: ', p)
        print('score: ', s)
        assert(s == score(p))
    
    # Save best result
    # TODO quickly put it back to quasi-steady state?
    # p = np.append(p, [-80, 5e2])
    params.append(best_p)
    scores.append(best_s)

    print('var: ', var)
    print('found best: ', best_p)
    print('score best: ', best_s)
    assert(best_s == score(best_p))

# Add them all up
all_p = np.array(params).reshape(len(params) * len(params[0]))
print('Final prt: ', all_p)
print('Total time: ', np.sum(all_p[1::2]))

#
# Store result
#
with open('%s/opt-prt-%s.txt' % (savedir, info.save_name), 'w') as f:
    f.write('# Voltage [mV]\tDuration [ms]\n')
    for i in range(len(all_p) // 2):
        f.write(pints.strfloat(all_p[2 * i]) \
                + '\t' \
                + pints.strfloat(all_p[2 * i + 1]) \
                + '\n' \
                )

print('Done')
