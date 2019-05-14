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

# step protocol
all_p = [-80, 200, 20, 500, -40, 500, -80, 200]

# Update protocol
model.set_voltage_protocol(all_p)
dt = 0.1
times = np.arange(0, np.sum(all_p[1::2]), dt)  # ms


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
last_t = 0
for i in range(len(all_p) // 6):
    p = all_p[6 * i:6 * (i + 1)]
    left_x = last_t + np.sum(p[1:-2:2])
    width_x = (np.sum(p[1::2]) - np.sum(p[1:-2:2]))
    plt.axvspan(left_x, left_x + width_x, facecolor="grey", alpha=0.2)
    last_t += np.sum(p[1::2])
plt.ylabel('Current (pA)')
plt.xlabel('Time (ms)')
plt.savefig('%s/test-prt-%s.png' % (savedir, info.save_name))


