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
base_param = np.array([ # Beattie et al. 2018 cell5
    1.52425200986726239e-01 * 1e3,  # g
    2.26083742297392306e-04,
    6.99218288065572152e-02,
    3.44967108766703847e-05,
    5.46117109135445741e-02,
    8.73172114272950939e-02,
    8.92915035388255028e-03,
    5.14870132009696314e-03,
    3.15632828645563066e-02,
    ]) # mV, ms

# Clamp ion; assumption in voltage clamp
ions_conc = {
    'potassium.Ki': 110,
    'potassium.Ko': 4,
    }  # mM

# Temperature of the experiment
temperature = 24.0  # oC

