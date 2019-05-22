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
import re

import model as m
import gen_samples

n_samples = 10

# Get all input variables
import importlib
# assume info files are in ./in
sys.path.append('../optimise-bruteforce/in')
try:
    info_id = sys.argv[1]
    info = importlib.import_module(info_id)
except:
    print('Usage: python %s [str:info_id]' \
            % os.path.basename(__file__))
    sys.exit()

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


# Load all protocols
allfiles = glob.glob('./converted_space_filling_designs/'
        + 'SpaceFillingProtocol[0-9][0-9].csv')

idxs = []
scores = []

def rmsd(t1, t2):
    return np.sqrt(np.mean((t1 - t2)**2))

for i, f in enumerate(allfiles):
    idxs.append(re.findall('\d+', f)[0])

    protocol = np.loadtxt(f, skiprows=1, delimiter=',')
    times = protocol[::5, 0]
    protocol = protocol[::5, 1]
    model.set_fixed_form_voltage_protocol(protocol, times)

    # Run simulations
    sims = []

    for i, p in enumerate(tested_parameter_samples):
        sims.append(model.simulate(p, times))
        if not np.all(np.isfinite(sims[-1])):
            scores.append(np.inf)
            continue

    # Compute differences
    total_diff = 0
    for i1 in range(len(tested_parameter_samples)):
        t1 = sims[i1]
        for i2 in range(i1 + 1, len(tested_parameter_samples)):
            t2 = sims[i2]
            total_diff += rmsd(t1, t2)

    score = -1. * total_diff / (0.5 * len(tested_parameter_samples) ** 2)

    scores.append(score)
    print(idxs[-1], scores[-1])

print('Best: SpaceFillingProtocol' + idxs[np.argmin(scores)])

with open('SpaceFillingProtocolRank-SamplingScore.csv', 'w') as f:
    f.write('\"Index\",\"score\"\n')
    for i, s in zip(idxs, scores):
        f.write(str(i) + ',' + str(s) + '\n')
