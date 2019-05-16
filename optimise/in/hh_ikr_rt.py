import numpy as np

# Output name
save_name = 'hh_ikr_rt'

# myokit mmt file path from `protocol-design/optimise`
model_file = '../mmt-model-files/beattie-2017-IKr.mmt'

# myokit current names that can be observed
# Assume only the sum of all current can be observed if multiple currents
current_list = ['ikr.IKr']

# myokit variable names
# All the parameters to be inferred
parameters = [
    'ikr.g',
    'ikr.p1', 'ikr.p2', 'ikr.p3', 'ikr.p4',
    'ikr.p5', 'ikr.p6', 'ikr.p7', 'ikr.p8',
    ]

# Indicating which current a parameter is belonged to
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

# Prior knowledge of the model parameters
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

# Clamp ion; assumption in voltage clamp
ions_conc = {
    'potassium.Ki': 110,
    'potassium.Ko': 4,
    }  # mM

# Prior
import sys
sys.path.append('../../lib/')  # where priors module is
from priors import HHIKrLogPrior as LogPrior

# Temperature of the experiment
temperature = 24.0  # oC

