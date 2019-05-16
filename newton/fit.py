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

"""
Run fit for single experiment-synthetic data study
"""

# Get all input variables
import importlib
sys.path.append('./in')  # assume info files are in ./in
sys.path.append('../optimise/in')  # assume info files are in ./in
try:
    prt_id = sys.argv[1]
    exp_id = sys.argv[2]
    cell = sys.argv[3]
    info_id = sys.argv[4]
    info = importlib.import_module(info_id)
except:
    print('Usage: python %s [str:protocol_id] [str:exp_id] [int:cell_id]' \
            % os.path.basename(__file__) + ' [str:info_id]' \
            + ' --optional [N_repeats]')
    sys.exit()

data_dir = './data'

savedir = 'out/%s-%s' % (info_id, prt_id)
if not os.path.isdir(savedir):
    os.makedirs(savedir)

temperature = 25.0
useFilterCap = False

# Set parameter transformation
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param

# Control fitting seed --> OR DONT
# fit_seed = np.random.randint(0, 2**30)
fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)


#
# Load data
#
data_file_name = exp_id + '-' + prt_id + '-' + cell + '.csv'
time_file_name = exp_id + '-' + prt_id + '-times.csv'
data = np.loadtxt(data_dir + '/' + data_file_name,
                  delimiter=',', skiprows=1) # headers
times = np.loadtxt(data_dir + '/' + time_file_name,
                   delimiter=',', skiprows=1) * 1e3 # headers; s -> ms
noise_sigma = np.std(data[:500])
print('Estimated noise level: ', noise_sigma)

protocol = np.loadtxt('protocol-time-series/protocol-' + prt_id + '.csv',
        delimiter=',', skiprows=1) # headers
skiptime = int((times[1] - times[0]) / (protocol[1, 0] - protocol[0, 0]))
protocol = protocol[::skiptime, 1]
protocol = protocol[:len(times)]
assert(len(protocol) == len(times))


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

# Update protocol
# fit full trace
prt_mask = None
model.set_fixed_form_voltage_protocol(protocol, times, prt_mask=prt_mask)

if useFilterCap:
    # Apply capacitance filter to data
    data = data * model.cap_filter(times)
if prt_mask is not None:
    # Apply protocol mask
    data = data * prt_mask


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
priorparams = np.asarray(info.base_param)
transform_priorparams = transform_from_model_param(priorparams)
print('Score at prior parameters: ',
        logposterior(transform_priorparams))
for _ in range(10):
    assert(logposterior(transform_priorparams) ==\
            logposterior(transform_priorparams))

# Run
try:
    N = int(sys.argv[5])
except IndexError:
    N = 3

params, logposteriors = [], []

for i in range(N):

    if False:  # i == 0:  # maybe not for syn data
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
                        pints.strfloat(info.base_param[k]))
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
obtained_parameters0 = params[0]
obtained_logposterior1 = logposteriors[1]
obtained_parameters1 = params[1]
obtained_logposterior2 = logposteriors[2]
obtained_parameters2 = params[2]

# Show results
print('Found solution:          Old parameters:' )
# Store output
with open('%s/solution-%s-%s-%s.txt' % (savedir, exp_id, cell, fit_seed), 
            'w') as f:
    for k, x in enumerate(obtained_parameters0):
        print(pints.strfloat(x) + '    ' + \
                pints.strfloat(info.base_param[k]))
        f.write(pints.strfloat(x) + '\n')
print('Found solution:          Old parameters:' )
# Store output
with open('%s/solution-%s-%s-%s-2.txt' % (savedir, exp_id, cell, fit_seed), 
            'w') as f:
    for k, x in enumerate(obtained_parameters1):
        print(pints.strfloat(x) + '    ' + \
                pints.strfloat(info.base_param[k]))
        f.write(pints.strfloat(x) + '\n')
print('Found solution:          Old parameters:' )
# Store output
with open('%s/solution-%s-%s-%s-3.txt' % (savedir, exp_id, cell, fit_seed), 
            'w') as f:
    for k, x in enumerate(obtained_parameters2):
        print(pints.strfloat(x) + '    ' + \
                pints.strfloat(info.base_param[k]))
        f.write(pints.strfloat(x) + '\n')

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
sol0 = problem.evaluate(transform_from_model_param(obtained_parameters0))
sol1 = problem.evaluate(transform_from_model_param(obtained_parameters1))
sol2 = problem.evaluate(transform_from_model_param(obtained_parameters2))
vol = model.voltage(times) * 1e3
axes[0].plot(times, vol, c='#7f7f7f')
axes[0].set_ylabel('Voltage [mV]')
axes[1].plot(times, data, alpha=0.5, label='data')
axes[1].plot(times, sol0, label='found solution 0')
axes[1].plot(times, sol1, label='found solution 1')
axes[1].plot(times, sol2, label='found solution 2')
axes[1].legend()
axes[1].set_ylabel('Current [pA]')
axes[1].set_xlabel('Time [s]')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/solution-%s-%s-%s.png' % (savedir, exp_id, cell, fit_seed),
        bbox_inches='tight')
plt.close()
