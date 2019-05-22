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
info_id = sys.argv[1]
info = importlib.import_module(info_id)
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
prior_parameters = np.asarray(info.prior_parameters)[:, 1:]  # only kinetics
for p in prior_parameters:
    p = np.append(1, p)
    plt.plot(test_t, model.simulate(p, test_t))
plt.savefig('%s/test-prior_parameters-%s' % (savedir, info.save_name))
plt.close()

# TODO transform to log space?
p_lower = np.amin(prior_parameters, axis=0)
p_upper = np.amax(prior_parameters, axis=0)
p_bound = pints.RectangularBoundaries(p_lower, p_upper)

parameter_samples = gen_samples.sample(p_bound, n=n_samples * 10)
parameter_samples = np.row_stack((prior_parameters, parameter_samples))
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
        parameter_samples=tested_parameter_samples,
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

for i_set in range(15): # TODO
    if i_set > 0:
        # Continue from previous simulation
        score.set_init_state(current_states)

    if i_set % 2:
        
        # Set random voltage and optimise duration
        sampled_param = full_boundaries.sample(1)[0]
        set_duration = None
        set_voltages = sampled_param[::2]
        x0 = sampled_param[1::2]
        score.set_duration(None)
        score.set_voltage(set_voltages)
        boundaries = pints.RectangularBoundaries(lower[1::2], upper[1::2])
        print('Optimising duration at voltage:', set_voltages)
        print('x0 (duration)', x0)
    else:
        # Set random duration and optimise voltage
        sampled_param = full_boundaries.sample(1)[0]
        set_voltages = None
        set_duration = sampled_param[1::2]
        x0 = sampled_param[::2]
        score.set_voltage(None)
        score.set_duration(set_duration)
        boundaries = pints.RectangularBoundaries(lower[::2], upper[::2])
        print('Optimising voltage at duration:', set_duration)
        print('x0 (voltage)', x0)

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
            out = np.zeros(score._n_parameters)
            if set_voltages is None:
                out[1::2] = set_duration
                out[::2] = p[:]
            elif set_duration is None:
                out[::2] = set_voltages
                out[1::2] = p[:]
            elif set_voltages is not None and set_duration is not None:
                out[:] = p[:]
            else:
                raise RuntimeError('Both voltage and duration are fixed.')
            params.append(out)
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
