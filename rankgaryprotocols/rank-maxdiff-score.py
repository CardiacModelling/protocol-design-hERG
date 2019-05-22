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

# Get all input variables
import importlib
# assume info files are in ./in
sys.path.append('../optimise-bruteforce/in')
try:
    info_id1 = sys.argv[1]
    info1 = importlib.import_module(info_id1)
    info_id2 = sys.argv[2]
    info2 = importlib.import_module(info_id2)
except:
    print('Usage: python %s [str:info_id1] [str:info_id2]' \
            % os.path.basename(__file__))
    sys.exit()

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

# Get parameters
p1 = info1.base_param
p2 = info2.base_param

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
    model1.set_fixed_form_voltage_protocol(protocol, times)
    model2.set_fixed_form_voltage_protocol(protocol, times)

    # Run simulations
    t1 = model1.simulate(p1, times)
    t2 = model2.simulate(p2, times)

    # Compute differences
    total_diff = rmsd(t1, t2)

    score = -1. * total_diff

    scores.append(score)
    print(idxs[-1], scores[-1])

print('Best: SpaceFillingProtocol' + idxs[np.argmin(scores)])

with open('SpaceFillingProtocolRank-MaxDiffScore.csv', 'w') as f:
    f.write('\"Index\",\"score\"\n')
    for i, s in zip(idxs, scores):
        f.write(str(i) + ',' + str(s) + '\n')
