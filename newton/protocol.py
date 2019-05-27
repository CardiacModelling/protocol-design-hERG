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

"""

"""

# Get all input variables
import importlib
sys.path.append('./in')  # assume info files are in ./in
sys.path.append('../optimise-bruteforce/in')
try:
    prt_id = sys.argv[1]
    info_id = sys.argv[2]
    info = importlib.import_module(info_id)
    info_list = []
    if len(sys.argv) > 3:
        for i in range(3, len(sys.argv)):
            info_list.append(importlib.import_module(sys.argv[i]))
except:
    print('Usage: python %s [str:protocol_id] [str:info_id] --optional' \
            % os.path.basename(__file__) + '[str:info_id_1, 2, ...]')
    sys.exit()

savedir = 'fig/protocols'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

temperature = 25.0
useFilterCap = False

# Set parameter transformation
transform_to_model_param = parametertransform.donothing
transform_from_model_param = parametertransform.donothing

protocol = np.loadtxt('protocol-time-series/protocol-' + prt_id + '.csv',
        delimiter=',', skiprows=1) # headers
times = protocol[:, 0]
protocol = protocol[:, 1]


#
# Do predictions
#
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 4))

# Model
model = m.Model(info.model_file,
        variables=info.parameters,
        current_readout=info.current_list,
        set_ion=info.ions_conc,
        temperature=273.15 + info.temperature,  # K
        transform=transform_to_model_param,
        )

# Update protocol
model.set_fixed_form_voltage_protocol(protocol, times, prt_mask=None)

# Simulate and plot
p = info.base_param
p[0] = p[0] / 2.

c = model.simulate(transform_from_model_param(p), times)
v = model.voltage(times)

axes[0].plot(times, v, c='#7f7f7f')
axes[1].plot(times, c, label=info.save_name)

for info_i in info_list:
    model = m.Model(info_i.model_file,
            variables=info_i.parameters,
            current_readout=info_i.current_list,
            set_ion=info_i.ions_conc,
            temperature=273.15 + info_i.temperature,  # K
            transform=transform_to_model_param,
            )

    model.set_fixed_form_voltage_protocol(protocol, times, prt_mask=None)

    p = info_i.base_param
    p[0] = p[0] / 2.
    c = model.simulate(transform_from_model_param(p), times)
    axes[1].plot(times, c, label=info_i.save_name)

axes[1].legend()
axes[0].set_ylabel('Voltage [mV]')
axes[1].set_ylabel('Current [pA]')
axes[1].set_xlabel('Time [ms]')
axes[1].axhline(0, color='#7f7f7f')
axes[1].set_ylim([-1150, 1350])
plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s.png' % (savedir, prt_id), bbox_inches='tight')
plt.close()
