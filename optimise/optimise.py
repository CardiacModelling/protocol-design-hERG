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

import model as m

savedir = 'out'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

save_name = 'hh-ikr'
model_file = '../mmt-model-files/beattie-2017-IKr.mmt'
current_list = ['ikr.IKr']
parameters = [
    'ikr.g',
    'ikr.p1', 'ikr.p2', 'ikr.p3', 'ikr.p4',
    'ikr.p5', 'ikr.p6', 'ikr.p7', 'ikr.p8',
    ]
var_list = {'ikr.g': 'ikr.IKr',
            'ikr.p1': 'ikr.IKr',
            'ikr.p2': 'ikr.IKr',
            'ikr.p3': 'ikr.IKr',
            'ikr.p4': 'ikr.IKr',
            'ikr.p5': 'ikr.IKr',
            'ikr.p6': 'ikr.IKr',
            'ikr.p7': 'ikr.IKr',
            'ikr.p8': 'ikr.IKr',
        }
base_param = np.array([ # herg25oc1 D19
    2.43765840715613886e+04,
    1.67935724044568219e-01,
    8.05876666209481272e+01,
    4.34335619635454820e-02,
    4.06775914336058904e+01,
    9.07128151348674265e+01,
    2.67085129074624632e+01,
    7.32142808226176189e+00,
    3.21575615314433136e+01,
    ]) * 1e-3  # V, s -> mV, ms
ions_conc = {
    'potassium.Ki': 110,
    'potassium.Ko': 4,
    }
temperature = 24.0

model = m.Model(model_file,
        variables=parameters,
        current_readout=current_list,
        set_ion=ions_conc,
        temperature=273.15 + temperature,  # K
        )

# protocol parameter: [step_1_voltage, step_1_duration, step_2..., step3...]
lower = [-120, 50,
        -120, 50,
        -120, 50,]
upper = [60, 1e3,
        60, 1e3,
        60, 1e3,]
boundaries = pints.RectangularBoundaries(lower, upper)

sensitivity = 0.025
params = []
scores = []
for i_var, var in enumerate(parameters):
    print('Optimising parameter %s in %s' % (var, var_list[var]))
    score = m.NaiveAllS1Score(model,
            max_idx=i_var,
            base_param=base_param,
            var_list=var_list,
            sensitivity=sensitivity)

    x0 = [-80, 5e2,
         -80, 5e2,
         -80, 5e2,]

    # Try it with x0
    print('Score at x0:', score(x0))
    for _ in range(10):
        assert(score(x0) == score(x0))
    
    opt = pints.OptimisationController(
            score,
            x0,
            boundaries=boundaries,
            method=pints.XNES)
    opt.set_max_iterations(None)
    opt.set_parallel(True)

    # Run optimisation
    try:
        with np.errstate(all='ignore'): # Tell numpy not to issue warnings
            p, s = opt.run()
            # TODO quickly put it back to quasi-steady state?
            # p = np.append(p, [-80, 5e2])
            params.append(p)
            scores.append(s)
    except ValueError:
        import traceback
        traceback.print_exc()

    print('var: ', var)
    print('found: ', p)
    print('score: ', s)
    print(score(p))

# Add them all up
all_p = np.array(params).reshape(len(params) * len(params[0]))
print('Final prt: ', all_p)
print('Total time: ', np.sum(all_p[1::2]))

#
# Store result
#
with open('%s/opt-prt-%s.txt' % (savedir, save_name), 'w') as f:
    f.write('# Voltage [mV]\tDuration [ms]\n')
    for i in range(len(all_p) // 2):
        f.write(pints.strfloat(all_p[2 * i]) \
                + '\t' \
                + pints.strfloat(all_p[2 * i + 1]) \
                + '\n' \
                )

'''
#
# Plot results
#
d = model.simulate(all_p)
total_c = []
for i in model.current_list():
    d[i] = d[i]
    total_c.append(d[i])
total_c = np.sum(total_c, axis=0)
v = model.voltage(all_p)
fig = plt.figure(figsize=(16, 6))
fig.add_subplot(211)
plt.plot(np.linspace(0, np.sum(all_p[1::2]), len(v)), v)
currentAxis = plt.gca()
last_t = 0
for i in range(len(all_p) // 6):
    p = all_p[6 * i:6 * (i + 1)]
    left_x = last_t + np.sum(p[1:-2:2])
    width_x = (np.sum(p[1::2]) - np.sum(p[1:-2:2]))
    currentAxis.add_patch(Rectangle((left_x, -250), width_x, 500, facecolor="grey", alpha=0.2))
    last_t += np.sum(p[1::2])
fig.add_subplot(212)
# plt.plot(t, d[model.current_list()[max_current]])
plt.plot(np.linspace(0, np.sum(all_p[1::2]), len(v)), total_c)
plt.ylim([-1, 2])
currentAxis = plt.gca()
last_t = 0
for i in range(len(all_p) // 6):
    p = all_p[6 * i:6 * (i + 1)]
    left_x = last_t + np.sum(p[1:-2:2])
    width_x = (np.sum(p[1::2]) - np.sum(p[1:-2:2]))
    currentAxis.add_patch(Rectangle((left_x, -250), width_x, 500, facecolor="grey", alpha=0.2))
    last_t += np.sum(p[1::2])
plt.savefig('%s/opt-prt.png' % savedir)

# Plot % of current
fig, axes = plt.subplots(len(variables), 1, figsize=(6, 12))
last_t = 0
for ii in range(len(all_p) // 6):
    p = all_p[6 * ii:6 * (ii + 1)]
    pre_time = (last_t + np.sum(p[1:-1:2]))# * 1e-3
    end_time = (last_t + np.sum(p[1::2]))# * 1e-3
    idx_i = int(pre_time / model._dt)
    idx_f = int(end_time / model._dt)
    # print(idx_i, idx_f)
    p_sensitivity = []
    for var in variables:
        cur = var_list[var]
        # print(var, cur)
        # p_i = p_i + sensitivity * p_i
        model.set_value_change(var, sensitivity)
        d1 = model.simulate(all_p, [cur])
        model.reset_value_change()
        # p_i = p_i - sensitivity * p_i
        model.set_value_change(var, -1 * sensitivity)
        d_1 = model.simulate(all_p, [cur])
        model.reset_value_change()
        dIdvar = (d1[cur][idx_i:idx_f] - d_1[cur][idx_i:idx_f]) \
                / (2 * sensitivity)
        p_sensitivity.append(np.sum(np.abs(dIdvar)))
    x = range(len(p_sensitivity))
    
    barlist = axes[ii].bar(x, p_sensitivity)
    barlist[ii].set_color('#e6550d')
    axes[ii].set_xticks([])
    axes[ii].set_ylim([np.min(p_sensitivity), np.max(p_sensitivity)])
    last_t += np.sum(p[1::2])
# plt.xticks(x, [i.split('.')[1] for i in model.current_list()])
plt.xticks(x, variables, rotation=90)
axes[len(all_p) // 6 // 2].set_ylabel('Percentage contribution')
plt.savefig('%s/matrix.png' % savedir)
'''
