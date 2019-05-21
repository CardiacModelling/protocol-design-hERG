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
import score_bruteforce as s
import gen_samples

savedir = './out'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

n_samples = 100

# Get all input variables
import importlib
sys.path.append('./in')  # assume info files are in ./in
try:
    info_id = sys.argv[1]
    info = importlib.import_module(info_id)
except:
    print('Usage: python %s [str:info_id]' \
            % os.path.basename(__file__))
    sys.exit()

n_repeats = 10

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

# Testing samples protocol
test_prt = [-80, 200, 20, 500, -40, 500, -80, 200]
test_t = np.arange(0, np.sum(test_prt[1::2]), 0.5)
model.set_voltage_protocol(test_prt)

# Generate parameter samples
prior_parameters = []  # only kinetics
for f in glob.glob('prior-parameters/%s-*/solution-*.txt' % info.save_name):
    p = np.loadtxt(f)
    p[0] = 1  # don't care too much about conductance; hopefully.
    prior_parameters.append(p[1:])
    plt.plot(test_t, model.simulate(p, test_t), c='C0' if 'A03' in f else 'C1')
plt.savefig('test')
plt.close()

# TODO transform to log space?
p_lower = np.amin(prior_parameters, axis=0)
p_upper = np.amax(prior_parameters, axis=0)
p_bound = pints.RectangularBoundaries(p_lower, p_upper)

parameter_samples = gen_samples.sample(p_bound, n=n_samples * 10)
# Add back (unity) conductance
parameter_samples = np.column_stack((np.ones(len(parameter_samples)),
                                     parameter_samples))

# Test samples
tested_parameter_samples = []
i = 0
model.set_voltage_protocol(test_prt)
from parametertransform import donothing
prior = info.LogPrior(donothing, donothing)
while len(tested_parameter_samples) < n_samples:
    if not np.isfinite(prior(parameter_samples[i])):
        print('Parameters outside bound...')
    elif not np.all(np.isfinite(model.simulate(parameter_samples[i], test_t))):
        print('Parameters not simulable...')
    else:
        tested_parameter_samples.append(parameter_samples[i])
    i += 1
    if i >= n_samples * 10:
        raise RuntimeError('Could not generate samples...') 

score = s.MaxParamDiffScore(model,
        parameters=tested_parameter_samples,
        )

x0 = [-80, 200, 20, 500, -40, 500]  # a pharma like protocol

if True:
    import timeit
    print('Single score evaluation time: %s s' \
        % (timeit.timeit(lambda: score(x0), number=10) / 10.))

sys.exit()

# Control fitting seed --> OR DONT
# fit_seed = np.random.randint(0, 2**30)
fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)

params, scores = [], []

for _ in range(n_repeats):

    for _ in range(15): # TODO
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
with open('%s/opt-prt-%s.txt' % (savedir, info.save_name), 'w') as f:
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
