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
    info_id1 = sys.argv[1]
    info1 = importlib.import_module(info_id1)
    info_id2 = sys.argv[2]
    info2 = importlib.import_module(info_id2)
except:
    print('Usage: python %s [str:info_id1] [str:info_id2]' \
            % os.path.basename(__file__))
    sys.exit()

n_repeats = 10

model1 = m.Model(info1.model_file,
        variables=info1.parameters,
        current_readout=info1.current_list,
        set_ion=info1.ions_conc,
        temperature=273.15 + info1.temperature,  # K
        )

model2 = m.Model(info2.model_file,
        variables=info2.parameters,
        current_readout=info2.current_list,
        set_ion=info2.ions_conc,
        temperature=273.15 + info2.temperature,  # K
        )

'''
x0 = [54, 1400, 0,
       26, 270, 0,
       10, 50, 0]  # square wave version of sine wave!
'''
x0 = np.loadtxt('out/a.txt')
x0 = x0.flatten()
#'''
t = np.arange(0, 2e3, 0.5)
model1.set_squarewave_voltage_protocol(x0, t)
model2.set_squarewave_voltage_protocol(x0, t)

v = np.loadtxt('../../nanion-data-export/protocol-time-series/protocol-staircaseramp.csv', delimiter=',', skiprows=1)
t = v[:, 0]
v = v[:, 1]
model1.set_fixed_form_voltage_protocol(v, t)
model2.set_fixed_form_voltage_protocol(v, t)

v1 = model1.voltage(t)
v2 = model2.voltage(t)
c1 = model1.simulate(info1.base_param, t)
c2 = model2.simulate(info2.base_param, t)
plt.subplot(211)
plt.plot(t, v1)
plt.plot(t, v2)
plt.subplot(212)
plt.plot(t, c1)
plt.plot(t, c2)
plt.savefig('%s/test' % savedir)

# Export
t = np.arange(0, 2e3, 0.1)
n_waves = int(len(x0) / 3)
v = np.zeros(t.shape) + -30  # TODO center at -30 mV?
for i in range(n_waves):
    v += m.squarewave(t, x0[3*i], x0[3*i+1], x0[3*i+2])
change_at = np.nonzero(np.append(v[:-1] - v[1:], 0))
step_duration = t[np.append(change_at, -1)] - np.append(0, t[change_at])
step_value = v[np.append(change_at, -1)]

np.savetxt('%s/export.txt' % savedir, np.array([step_value, step_duration]).T)
