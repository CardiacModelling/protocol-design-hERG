import numpy as np

# Output name
save_name = 'wang_ikr_rt'

# myokit mmt file path from `protocol-design/optimise`
model_file = '../mmt-model-files/wang-ikr.mmt'

# myokit current names that can be observed
# Assume only the sum of all current can be observed if multiple currents
current_list = ['ikr.IKr']

# myokit variable names
# All the parameters to be inferred
parameters = [
    'ikr.g',
    'ikr.p1', 'ikr.p2', 'ikr.p3', 'ikr.p4',
    'ikr.p5', 'ikr.p6', 'ikr.p7', 'ikr.p8',
    'ikr.p9', 'ikr.p10', 'ikr.p11', 'ikr.p12',
    'ikr.p13', 'ikr.p14',
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
            'ikr.p9': 'ikr.IKr',
            'ikr.p10': 'ikr.IKr',
            'ikr.p11': 'ikr.IKr',
            'ikr.p12': 'ikr.IKr',
            'ikr.p13': 'ikr.IKr',
            'ikr.p14': 'ikr.IKr',
        }

# Prior knowledge of the model parameters
base_param = np.array([ # Beattie et al. 2018 cell5
        2.11451916137530976e-01 * 1e3,  # g
        3.40480255129870294e-04,
        1.09853080133853545e-01,
        2.78158636962794539e-03,
        1.06603162448445768e-01,
        2.34436215593156411e-03,
        1.04705086663227821e-07,
        3.15078572900222374e-04,
        3.99208131967221866e-02,
        1.23597674742260924e-01,
        1.55514220791980115e-02,
        5.74918399655915176e-03,
        2.89464023164944483e-02,
        1.30731037065575720e-02,
        6.74786425362727382e-03,
    ]) #* 1e-3  # V, s -> mV, ms

# Clamp ion; assumption in voltage clamp
ions_conc = {
    'potassium.Ki': 120,
    'potassium.Ko': 5,
    }  # mM

# Prior
import sys
sys.path.append('../../lib/')  # where priors module is
#from priors import WangIKrLogPrior as LogPrior

# Prior parameters
import glob
prior_parameters = []
prior_parameters.append(base_param)
idx_map_dom2sanmitra = np.array(
        [11, 12, 13, 14, 1, 2, 3, 4, 7, 8, 9, 10, 5, 6]) - 1  # Python idx
""" Sanmitra's model-16 parameters -> Dom's Wang-1997-model parameters
a1 = k51
1,2 -> 11,12
b1 = k15
3,4 -> 13,14
a2 = k12
13 -> 5
b2 = k21
14 -> 6
a3 = k23
5,6 -> 1,2
b3 = k32
7,8 -> 3,4
a4 = k34
9,10 -> 7,8
b4 = k43
11,12 -> 9,10
"""
for f in glob.glob(
        './%s-prior-parameters/cell-*-fit-*-parameters-[0-1][0-9].txt' \
        % save_name):
    p = np.loadtxt(f)
    g = np.mean(p[-3:])  # Just how Dom arranged it, last 3 are conductance...
    p = p[:-3][idx_map_dom2sanmitra]  # Map Dom's parameters to Sanmitra's ones
    prior_parameters.append(np.append(g, p))

# Temperature of the experiment
temperature = 24.0  # oC

