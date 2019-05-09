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
for i in range(len(all_p) // 6):
    p = all_p[6 * i:6 * (i + 1)]
    left_x = last_t + np.sum(p[1:-2:2])
    width_x = (np.sum(p[1::2]) - np.sum(p[1:-2:2]))
    plt.axvspan(left_x, left_x + width_x, facecolor="grey", alpha=0.2)
    last_t += np.sum(p[1::2])
plt.ylabel('Voltage (mV)')
fig.add_subplot(212)
# plt.plot(times, d[model.current_list()[max_current]]) # show specific current
plt.plot(times, total_c)
plt.ylim([-500, 500])
last_t = 0
for i in range(len(all_p) // 6):
    p = all_p[6 * i:6 * (i + 1)]
    left_x = last_t + np.sum(p[1:-2:2])
    width_x = (np.sum(p[1::2]) - np.sum(p[1:-2:2]))
    plt.axvspan(left_x, left_x + width_x, facecolor="grey", alpha=0.2)
    last_t += np.sum(p[1::2])
plt.ylabel('Current (pA)')
plt.xlabel('Time (ms)')
plt.savefig('%s/opt-prt-%s.png' % (savedir, info.save_name))


# Plot % of current
fig, axes = plt.subplots(len(info.parameters), 1, figsize=(6, 12))
last_t = 0
for ii in range(len(all_p) // 6):
    p = all_p[6 * ii:6 * (ii + 1)]
    pre_time = (last_t + np.sum(p[1:-1:2]))# * 1e-3
    end_time = (last_t + np.sum(p[1::2]))# * 1e-3
    idx_i = int(pre_time / dt)
    idx_f = int(end_time / dt)

    p_sensitivity = []
    for i, var in enumerate(info.parameters):
        cur = info.var_list[var]

        # p_i = p_i + sensitivity * p_i
        x = np.copy(info.base_param)
        x[i] = x[i] * (1 + sensitivity)
        d1 = model.simulate(x, times, [cur])
        # p_i = p_i - sensitivity * p_i
        x = np.copy(info.base_param)
        x[i] = x[i] * (1 - sensitivity)
        d_1 = model.simulate(x, times, [cur])

        dIdvar = (d1[cur][idx_i:idx_f] - d_1[cur][idx_i:idx_f]) \
                / (2 * sensitivity)
        p_sensitivity.append(np.sum(np.abs(dIdvar)) / len(dIdvar))
    x = range(len(p_sensitivity))

    barlist = axes[ii].bar(x, p_sensitivity)
    barlist[ii].set_color('#e6550d')
    axes[ii].set_xticks([])
    axes[ii].set_ylim([np.min(p_sensitivity), np.max(p_sensitivity)])
    last_t += np.sum(p[1::2])

# plt.xticks(x, [i.split('.')[1] for i in model.current_list()])
plt.xticks(x, info.parameters, rotation=90)
axes[len(all_p) // 6 // 2].set_ylabel('Parameter sensitivity')
plt.tight_layout()
plt.savefig('%s/opt-prt-%s-matrix.png' % (savedir, info.save_name))

