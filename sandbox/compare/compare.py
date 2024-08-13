#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('../lib/')  # where the lib are
sys.path.append('./in/')  # where the model def are
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model as m
import parametertransform

import glob
import importlib

"""
Compare models under a protocol simulation.
"""

# Get all input variables
try:
    prt_file = sys.argv[1]
    prt_id = os.path.splitext(os.path.basename(prt_file))[0]
except:
    print('Usage: python %s [str:protocol_dir]' % os.path.basename(__file__))
    sys.exit()

savedir = './out'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

temperature = 25.0
useFilterCap = False

# Set parameter transformation
transform_to_model_param = parametertransform.donothing
transform_from_model_param = parametertransform.donothing


# Load protocol
all_p = np.loadtxt(prt_file)
# reshape it to [step_1_voltage, step_1_duration, ...]
all_p = all_p.flatten()

# Time
dt = 0.5
times = np.arange(0, np.sum(all_p[1::2]), dt)  # ms

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

#
# Go through different models
#
info_files = glob.glob('./in/*.py')
for info_file in info_files:
    info_id = os.path.splitext(os.path.basename(info_file))[0]
    info = importlib.import_module(info_id)

    # Model
    model = m.Model(info.model_file,
            variables=info.parameters,
            current_readout=info.current_list,
            set_ion=info.ions_conc,
            temperature=273.15 + info.temperature,  # K
            transform=transform_to_model_param,
            )

    prt_mask = None
    model.set_voltage_protocol(all_p, prt_mask=prt_mask)

    parameters = np.copy(info.base_param)

    # Simulation
    sim = model.simulate(parameters, times)
    vol = model.voltage(times)

    if useFilterCap:
        # Apply capacitance filter to data
        sim = sim * model.cap_filter(times)
    if prt_mask is not None:
        # Apply protocol mask
        sim = sim * prt_mask

    # Plot
    axes[0].plot(times, vol, c='#7f7f7f')
    axes[1].plot(times, sim, label=info_id)

axes[0].set_ylabel('Voltage [mV]')
axes[1].set_ylabel('Current [pA]')
axes[1].set_xlabel('Time [s]')
axes[1].legend()
plt.subplots_adjust(hspace=0)
plt.savefig('%s/compare-%s.png' % (savedir, prt_id), bbox_inches='tight')
plt.close()
