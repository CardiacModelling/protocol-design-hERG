import os
import sys
import numpy as np

# Get all input variables
import importlib
sys.path.append('./in')  # assume info files are in ./in
info_id = sys.argv[1]
info = importlib.import_module(info_id)

prior_parameters = np.asarray(info.prior_parameters)[:, 1:]  # only kinetics
for p in prior_parameters:
    p = np.append(1, p)

# TODO transform to log space?
p_lower = np.amin(prior_parameters, axis=0)
p_upper = np.amax(prior_parameters, axis=0)

print(p_lower)
print(p_upper)
