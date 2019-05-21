#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('../lib/')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model as m
import parametertransform

import glob
import re

"""

"""

# Get all input variables
import importlib
sys.path.append('./in')  # assume info files are in ./in
sys.path.append('../optimise-localsensitivity/in')
try:
    prt_id = sys.argv[1]
    exp_id = sys.argv[2]
    cell = sys.argv[3]
    info_id = sys.argv[4]
    info = importlib.import_module(info_id)
except:
    print('Usage: python %s [str:protocol_id] [str:exp_id] [int:cell_id]' \
            % os.path.basename(__file__) + ' [str:info_id]')
    sys.exit()

data_dir = './data'

savedir = 'fig/%s' % (info_id)
if not os.path.isdir(savedir):
    os.makedirs(savedir)

temperature = 25.0
useFilterCap = False

# Set parameter transformation
transform_to_model_param = parametertransform.donothing
transform_from_model_param = parametertransform.donothing

# For getting data back
fit_seed = 542811797

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

protocol = np.loadtxt('protocol-time-series/protocol-' + prt_id + '.csv',
        delimiter=',', skiprows=1) # headers
skiptime = int((times[1] - times[0]) / (protocol[1, 0] - protocol[0, 0]))
protocol = protocol[::skiptime, 1]
protocol = protocol[:len(times)]
assert(len(protocol) == len(times))


#
# Do predictions
#
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# Model
model = m.Model(info.model_file,
        variables=info.parameters,
        current_readout=info.current_list,
        set_ion=info.ions_conc,
        temperature=273.15 + info.temperature,  # K
        transform=transform_to_model_param,
        )

# Data
axes[1].plot(times, data, alpha=0.5, label='data')

# Update protocol
model.set_fixed_form_voltage_protocol(protocol, times, prt_mask=None)

folders = glob.glob('./out/' + info_id + '-*')
for folder in folders:
    if not os.path.isdir(folder):
        continue

    prt = re.findall('./out/' + info_id + '-(.*)', folder)[0]

    # Extract best 3
    p = np.loadtxt(folder + '/solution-%s-%s-%s.txt' \
            % (exp_id, cell, fit_seed))

    c = model.simulate(transform_from_model_param(p), times)

    v = model.voltage(times)

    axes[0].plot(times, v, c='#7f7f7f')

    axes[1].plot(times, c, label=prt)

axes[1].legend()
axes[0].set_ylabel('Voltage [mV]')
axes[1].set_ylabel('Current [pA]')
axes[1].set_xlabel('Time [s]')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s-%s-%s-%s.png' % (savedir, prt_id, info_id, exp_id, cell),
        bbox_inches='tight')
plt.close()
