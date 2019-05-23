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

savedir = './out'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Get all input variables
import importlib
sys.path.append('./in')  # assume info files are in ./in
try:
    info_id = sys.argv[1]
    info = importlib.import_module(info_id)
except:
    print('Usage: python %s [str:info_id]' % os.path.basename(__file__))
    sys.exit()

model = m.Model(info.model_file,
        variables=info.parameters,
        current_readout=info.current_list,
        set_ion=info.ions_conc,
        temperature=273.15 + info.temperature,  # K
        )

# Load protocol
all_p = np.loadtxt('%s/opt-prt-%s.txt' % (savedir, info.save_name))
# reshape it to [step_1_voltage, step_1_duration, ...]
all_p = all_p.flatten()

# Update protocol
model.set_voltage_protocol(all_p)
dt = 0.1
times = np.arange(0, np.sum(all_p[1::2]), dt)  # ms

sensitivity = 0.025

#
# Plot results
#
total_c = model.simulate(info.base_param, times)
v = model.voltage(times)
fig = plt.figure(figsize=(16, 6))
fig.add_subplot(211)
plt.plot(times, v)
last_t = 0
for i in range(0, len(all_p) // 6):
    left_x = last_t + np.sum(all_p[12 * i: 12 * i + 6][1::2])
    width_x = np.sum(all_p[12 * i + 6:12 * (i + 1)][1::2])
    plt.axvspan(left_x, left_x + width_x, facecolor="grey", alpha=0.2)
    last_t += np.sum(all_p[12 * i:12 * (i + 1)][1::2])
plt.ylabel('Voltage (mV)')
fig.add_subplot(212)
# plt.plot(times, d[model.current_list()[max_current]]) # show specific current
plt.plot(times, total_c)
plt.axhline(0, color='#7f7f7f')
# plt.ylim([-500, 500])
last_t = 0
for i in range(0, len(all_p) // 6):
    left_x = last_t + np.sum(all_p[12 * i: 12 * i + 6][1::2])
    width_x = np.sum(all_p[12 * i + 6:12 * (i + 1)][1::2])
    plt.axvspan(left_x, left_x + width_x, facecolor="grey", alpha=0.2)
    last_t += np.sum(all_p[12 * i:12 * (i + 1)][1::2])
plt.ylabel('Current (pA; g=%.1fpA/mV)' % info.base_param[0])
plt.xlabel('Time (ms)')
plt.savefig('%s/opt-prt-%s.png' % (savedir, info.save_name))



