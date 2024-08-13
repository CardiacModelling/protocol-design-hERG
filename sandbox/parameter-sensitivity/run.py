#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model as m

from SALib.sample import saltelli
from SALib.analyze import sobol

savedir = './out'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# This is giving n_samples * (n_parameters + 2) number of evaluations
n_samples = 5

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
# Control fitting seed --> OR DONT
fit_seed = np.random.randint(0, 2**30)
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)

model = m.Model(info.model_file,
        variables=info.parameters,
        current_readout=info.current_list,
        set_ion=info.ions_conc,
        temperature=273.15 + info.temperature,  # K
        )

# Testing samples protocol
test_prt = [-80, 200, 20, 500, -40, 500, -80, 200]
test_t = np.arange(0, np.sum(test_prt[1::2]), 0.5)
model.set_voltage_protocol(test_prt)

# Generate parameter samples
prior_parameters = np.asarray(info.prior_parameters)
n_prior_param = len(prior_parameters)
prior_parameters = [prior_parameters[0]] * n_prior_param
prior_parameters[1:] = [(2. ** np.random.uniform(-1, 1, size=len(p))) * p
        for p in prior_parameters[1:]]
for p in prior_parameters:
    test_p = np.append(1, p[1:])
    plt.plot(test_t, model.simulate(test_p, test_t))
plt.savefig('%s/test-prior_parameters-%s' % (savedir, info.save_name))
plt.close()

# TODO transform to log space?
# p_lower = np.amin(prior_parameters, axis=0)
# p_upper = np.amax(prior_parameters, axis=0)
p_lower = prior_parameters[0] / 2.
p_upper = prior_parameters[0] * 2.
print('Lower bound: ', p_lower)
print('Upper bound: ', p_upper)

# Test samples
def test_function(model, p, samples=None):
    # Test if function
    from parametertransform import donothing
    prior = info.LogPrior(donothing, donothing)
    if samples is not None:
        return prior.sample(samples)

    test_prt = [-80, 200, 20, 500, -40, 500, -80, 200]
    test_t = np.arange(0, np.sum(test_prt[1::2]), 0.5)
    model.set_voltage_protocol(test_prt)

    out = []
    for p_i in p:
        if not np.isfinite(prior(p_i)):
            print('Parameters outside bound...')
            print(p_i)
            out.append(False)
        elif not np.all(np.isfinite(model.simulate(p_i, test_t))):
            print('Parameters not simulable...')
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

parameter_samples = saltelli.sample(salib_problem, n_samples,
        calc_second_order=False)

passed = test_function(model, parameter_samples)
print('There are %s out of %s parameters passed the test.' \
        % (np.sum(passed), len(parameter_samples)))

if np.sum(passed) < len(parameter_samples):
    target_parameter_samples = len(parameter_samples)
    ratio = target_parameter_samples / np.float(np.sum(passed))
    print('Resampling with %.2f times more samples.' % ratio)
    parameter_samples = saltelli.sample(salib_problem,
            int(ratio * n_samples), calc_second_order=False)
    passed = test_function(model, parameter_samples)

    if np.sum(passed) / target_parameter_samples > 1.1:
        # Don't want to remove too much... will/might introduce
        # bias in the samples?!
        raise Exception('Wait...')

    elif np.sum(passed) > target_parameter_samples:
        n_extra = np.sum(passed) - target_parameter_samples
        print('Randomly removing %s extra samples.' % n_extra)
        passed_samples = parameter_samples[passed]
        np.random.shuffle(passed_samples)
        parameter_samples = passed_samples[:-1 * n_extra]

    elif np.sum(passed) < len(parameter_samples):
        n_missing = target_parameter_samples - np.sum(passed)
        print('Filling %s samples with uniformly sampled ' \
                % n_missing + 'parameters.')
        addition = test_function(None, None, samples=n_missing)
        parameter_samples = np.row_stack(
                (parameter_samples[passed], addition))

# Assessing model protocol
assess_prt = np.loadtxt('../optimise-sobol/out/opt-prt-%s.txt' %
        info.save_name)
# reshape it to [step_1_voltage, step_1_duration, ...]
assess_prt = assess_prt.flatten()
assess_t = np.arange(0, np.sum(assess_prt[1::2]), 0.5)
model.set_voltage_protocol(assess_prt)
    
def assess_function(trace):
    return np.mean(trace ** 2)

f_sims = []
for i, p in enumerate(parameter_samples):
    # Only use a scaler measure of the simulation output
    sim = model.simulate(p, assess_t)
    plt.plot(assess_t, sim)
    f_sims.append(assess_function(sim))
    if not np.any(np.isfinite(f_sims[-1])):
        raise RuntimeError('Not here')
plt.savefig('%s/assess-prior_parameters-%s' % (savedir, info.save_name))

# Compute Sobol sensitivity
Si = sobol.analyze(salib_problem, np.asarray(f_sims), calc_second_order=False)

for i in range(model.n_parameters()):
    sobol_t = Si['ST'][i]
    print(i, sobol_t)


print('Done')
