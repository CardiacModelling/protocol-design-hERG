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
import glob

import model as m
import score_sobol as s
import gen_samples

savedir = './out'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

n_samples = 100  # giving n_samples * (n_parameters + 2) number of evaluations

# Get all input variables
import importlib
sys.path.append('./in')  # assume info files are in ./in
try:
    info_id = sys.argv[1]
    info = importlib.import_module(info_id)
except:
    print('Usage: python %s [str:info_id] --optional [int:seed]' \
            % os.path.basename(__file__))
    sys.exit()

try:
    seed_id = int(sys.argv[2])
except:
    seed_id = 101
np.random.seed(seed_id)
print('Seed ID: ', seed_id)

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
full_boundaries = pints.RectangularBoundaries(lower, upper)

# Testing samples protocol
test_prt = [-80, 200, 20, 500, -40, 500, -80, 200]
test_t = np.arange(0, np.sum(test_prt[1::2]), 0.5)
model.set_voltage_protocol(test_prt)

# Generate parameter samples
prior_parameters = np.asarray(info.prior_parameters)
for p in prior_parameters:
    test_p = np.append(1, p[1:])
    plt.plot(test_t, model.simulate(test_p, test_t))
plt.savefig('%s/test-prior_parameters-%s' % (savedir, info.save_name))
plt.close()

# TODO transform to log space?
p_lower = np.amin(prior_parameters, axis=0)
p_upper = np.amax(prior_parameters, axis=0)

# Test samples
def test_function(model, p):
    # Test if function
    from parametertransform import donothing
    test_prt = [-80, 200, 20, 500, -40, 500, -80, 200]
    test_t = np.arange(0, np.sum(test_prt[1::2]), 0.5)
    model.set_voltage_protocol(test_prt)
    prior = info.LogPrior(donothing, donothing)

    out = []
    for p_i in p:
        if not np.isfinite(prior(p_i)):
            # print('Parameters outside bound...')
            out.append(False)
        elif not np.all(np.isfinite(model.simulate(p_i, test_t))):
            # print('Parameters not simulable...')
            out.append(False)
        else:
            out.append(True)
    return out

# Problem setting for package SALib
salib_problem = {
    'num_vars': model.n_parameters(),
    'names': ['p%s' % (i) for i in range(model.n_parameters())],
    'bounds': np.array([p_lower, p_upper]).T,
}

score = s.MaxSobolScore(model,
        problem=salib_problem,
        max_idx=0,  # to be iterated through
        test_func=test_function,
        n=n_samples,
        )
score.set_init_state(None)

x0 = [-80, 200, 20, 500, -40, 500]  # a pharma like protocol

if True:
    import timeit
    print('Single score evaluation time: %s s' \
        % (timeit.timeit(lambda: score(x0), number=10) / 10.))

# Control fitting seed --> OR DONT
fit_seed = np.random.randint(0, 2**30)
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)

params, scores = [], []

for i_set in range(model.n_parameters()):
    if i_set > 0:
        # Continue from previous simulation
        score.set_init_state(current_states)
    else:
        # Start from steady state
        score.set_init_state(None)

    # Set a different model parameter to be maximised
    score.set_max_idx(i_set)

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
            print('Found solution:' )
            for k, x in enumerate(p):
                print(pints.strfloat(x))
    except ValueError:
        import traceback
        traceback.print_exc()
        raise RuntimeError('Not here...')

    # Get current state
    current_states = score(p, get_state=True)

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
