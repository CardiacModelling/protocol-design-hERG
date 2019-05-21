#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

expid = 'newtonrun1'
prtids = ['staircaseramp', 'sis', 'hh3step', 'wang3step']
cell = 'A13'
infoid = 'hh_ikr_rt'
seed = '542811797'
markers = ['o', 's', '^', 'd']


labels = [r'$g$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
               r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$']

fig = plt.figure(figsize=(8, 4))

for i_prt, prtid in enumerate(prtids):
    folder = './out/' + infoid + '-' + prtid
    for i in range(3):
        if i == 0:
            p = np.loadtxt('%s/solution-newtonrun1-%s-542811797.txt' \
                    % (folder, cell))
        else:
            p = np.loadtxt('%s/solution-newtonrun1-%s-542811797-%s.txt' \
                    % (folder, cell, i + 1))
        plt.plot(range(len(p)), np.log10(p), ls='', marker=markers[i_prt],
                c='C' + str(i_prt), label='__nolegend__' if i else prtid)

plt.legend()
plt.title('Model: %s; Exp: %s; Cell: %s' % (infoid, expid, cell))
plt.ylabel(r'$\log_{10}$(value)')
plt.xticks(range(len(p)), labels)
plt.savefig('out/%s-%s-%s' % (infoid, expid, cell))

