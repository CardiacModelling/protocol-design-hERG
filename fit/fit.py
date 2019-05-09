#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('../lib/')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pints

import model as m
import parametertransform
from priors import HHIKrLogPrior as LogPrior
from priors import prior_parameters

"""
Run fit for single experiment-synthetic data study
"""

# Get all input variables
import importlib
sys.path.append('./in')  # assume info files are in ./in
sys.path.append('../optimise/in')  # assume info files are in ./in
try:
    prt_file = sys.argv[1]
    info_id = sys.argv[2]
    info = importlib.import_module(info_id)
    cell = sys.argv[3]
    int(cell)
except:
    print('Usage: python %s [str:protocol_dir] [str:info_id] [int:cell_id]' \
            % os.path.basename(__file__) + ' --optional [N_repeats]')
    sys.exit()

cov_seed = 101

savedir = 'out/syn-%s' % (cov_seed)
if not os.path.isdir(savedir):
    os.makedirs(savedir)
savetruedir = 'out/syn-%s-true' % (cov_seed)
if not os.path.isdir(savetruedir):
    os.makedirs(savetruedir)

temperature = 25.0
useFilterCap = False

# Set parameter transformation
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param


#
# Set true value
#

path2dir = '../../hERGRapidCharacterisation/'

fakedatanoise = 11.0  # roughly what the recordings are, 10-12 pA
path2mean = path2dir + 'room-temperature-only/kylie-room-temperature/' \
            + 'last-solution_C5.txt'
mean = np.loadtxt(path2mean)
mean[1:] = mean[1:] * 1e-3  # V, s -> mV, ms
std = np.array([0.41290891, 0.64938737, 0.15619636, 0.44034459, 0.10485176,
        0.19219367, 0.30296862, 0.19692015, 0.07103704])  # from hERG paper 1
assert(len(std) == len(mean))
# Transform parameter to search space
mean = transform_from_model_param(mean)

# Give some funny correlation for parameters
import sklearn.datasets
# rho = sklearn.datasets.make_spd_matrix(len(stddev), random_state=1)
corr = sklearn.datasets.make_sparse_spd_matrix(len(std), alpha=0.25,
        norm_diag=True, random_state=cov_seed)
std_ = np.asanyarray(std)
covariance = corr * np.outer(std_, std_)

# Save it too
np.savetxt('./out/corr-%s.txt' % cov_seed, corr)
np.savetxt('./out/cov-%s.txt' % cov_seed, covariance)

# Control fitting seed --> OR DONT
np.random.seed(int(cell))
fit_seed = np.random.randint(0, 2**30)
print('Using seed: ', fit_seed)
np.random.seed(fit_seed)

parameters = np.random.multivariate_normal(mean, covariance)


#
# Store true parameters
#

with open('%s/solution-%s.txt' % (savetruedir, cell), 'w') as f:
    for x in transform_to_model_param(parameters):
        f.write(pints.strfloat(x) + '\n')


#
# Generate syn. data
#

# Model
model = m.Model(info.model_file,
        variables=info.parameters,
        current_readout=info.current_list,
        set_ion=info.ions_conc,
        temperature=273.15 + info.temperature,  # K
        transform=transform_to_model_param,
        )

# Load protocol
all_p = np.loadtxt(prt_file)
# reshape it to [step_1_voltage, step_1_duration, ...]
all_p = all_p.flatten()

# Update protocol
prt_mask = None  # TODO cover up part of the measurement
model.set_voltage_protocol(all_p, prt_mask=prt_mask)
dt = 0.5
times = np.arange(0, np.sum(all_p[1::2]), dt)  # ms

# generate from model + add noise
data = model.simulate(parameters, times)
data += np.random.normal(0, fakedatanoise, size=data.shape)
if useFilterCap:
    # Apply capacitance filter to data
    data = data * model.cap_filter(times)

# Estimate noise from start of data
noise_sigma = np.std(data[:200])


#
# Fit
#

# Create Pints stuffs
problem = pints.SingleOutputProblem(model, times, data)
loglikelihood = pints.KnownNoiseLogLikelihood(problem, noise_sigma)
logprior = LogPrior(transform_to_model_param,
                    transform_from_model_param)
logposterior = pints.LogPosterior(loglikelihood, logprior)

# Check logposterior is working fine
priorparams = np.asarray(prior_parameters['23.0'])
transform_priorparams = transform_from_model_param(priorparams)
print('Score at prior parameters: ',
        logposterior(transform_priorparams))
for _ in range(10):
    assert(logposterior(transform_priorparams) ==\
            logposterior(transform_priorparams))

# Run
try:
    N = int(sys.argv[4])
except IndexError:
    N = 3

params, logposteriors = [], []

for i in range(N):

    if i == 0:
        x0 = transform_priorparams
    else:
        # Randomly pick a starting point
        x0 = logprior.sample()
    print('Starting point: ', x0)

    # Create optimiser
    print('Starting logposterior: ', logposterior(x0))
    opt = pints.Optimisation(logposterior, x0.T, method=pints.CMAES)
    opt.set_max_iterations(None)
    opt.set_parallel(False)

    # Run optimisation
    try:
        with np.errstate(all='ignore'):
            # Tell numpy not to issue warnings
            p, s = opt.run()
            p = transform_to_model_param(p)
            params.append(p)
            logposteriors.append(s)
            print('Found solution:          Old parameters:' )
            for k, x in enumerate(p):
                print(pints.strfloat(x) + '    ' + \
                        pints.strfloat(priorparams[k]))
    except ValueError:
        import traceback
        traceback.print_exc()

# Order from best to worst
order = np.argsort(logposteriors)[::-1]  # (use [::-1] for LL)
logposteriors = np.asarray(logposteriors)[order]
params = np.asarray(params)[order]

# Show results
bestn = min(3, N)
print('Best %d logposteriors:' % bestn)
for i in xrange(bestn):
    print(logposteriors[i])
print('Mean & std of logposterior:')
print(np.mean(logposteriors))
print(np.std(logposteriors))
print('Worst logposterior:')
print(logposteriors[-1])

# Extract best 3
obtained_logposterior0 = logposteriors[0]
obtained_parameters = params[0]

# Show results
print('Found solution:          Old parameters:' )
# Store output
with open('%s/solution-%s.txt' % (savedir, cell), 'w') as f:
    for k, x in enumerate(obtained_parameters):
        print(pints.strfloat(x) + '    ' + \
                pints.strfloat(priorparams[k]))
        f.write(pints.strfloat(x) + '\n')

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
sol = problem.evaluate(transform_from_model_param(obtained_parameters))
vol = model.voltage(times) * 1e3
axes[0].plot(times, vol, c='#7f7f7f')
axes[0].set_ylabel('Voltage [mV]')
axes[1].plot(times, data, alpha=0.5, label='data')
axes[1].plot(times, sol, label='found solution')
axes[1].legend()
axes[1].set_ylabel('Current [pA]')
axes[1].set_xlabel('Time [s]')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/solution-%s.png' % (savedir, cell), bbox_inches='tight')
plt.close()
