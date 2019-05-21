#This script can be used to plot IV curves (simulation and experiment)
#Author: Dominic Whittaker
#Date: 8th January 2015
#Compile with python plot_IV_IKr.py
#See 'plot.py' for original template script

import numpy as np
import matplotlib.pyplot as p
from matplotlib import *

p.rcdefaults()

p.rc('lines', markeredgewidth=2)
p.rc('lines', markersize=7)
p.rc('xtick.major', size=5)
p.rc('ytick.major', size=5) #changes size of y axis ticks on both sides
p.rc('xtick', direction='out')
p.rc('ytick', direction='out')

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return p.cm.get_cmap(name, n)

n_runs = 20
n_params = 17
a = np.zeros((n_runs,n_params))
cmap = get_cmap(n_runs)

for i in range(n_runs):
	filename = 'cell-10-fit-3-parameters-' + str(i+1) + '.txt'
	b = np.loadtxt(filename, unpack=True )
	a[i] = b

x_true = np.array([ # WT
2.28337972778078317e-02,
9.51931261334773397e-03,
9.27464602276671809e-04,
2.98296335849273959e-02,
8.94469830945009842e-03,
1.05927986028410102e-02,
2.10871802606201725e-01,
2.11159486150373082e-02,
3.66902379249811617e-03,
2.82899801440527447e-02,
1.25245721023515971e-02,
7.65758245825652328e-02,
7.29927925185385151e-06,
1.56129065201256023e-01,
3.60847724727575370e-01,
1.98085272663462902e-01,
7.51523424080540736e-02
	])

# x_true = np.array([ # R56Q
#     7.46102348817899164e-02,
#     1.65999164287386758e-02,
#     1.63620294971953649e-02,
#     1.33664327613370326e-02,
#     6.51823757107755294e-03,
#     3.33578013480349683e-02,
#     1.39020523437499988e-01,
#     3.16568501726329707e-02,
#     1.05496712801113612e-02,
#     2.77032867567961402e-02,
#     2.23028801744423778e-02,
#     8.58249140862797910e-02,
#     7.03466397988902911e-07,
#     1.53309008982847300e-01,
#     2.14788251887469112e-01,
#     2.12124481253535502e-01,
#     5.05302860579234236e-02
# ])

fig = p.figure(1, figsize=(7,5),dpi=200) #(8,6) seems to be the default
fig.set_facecolor('white') #this changes the background colour from the default grey/blue to white

ax1 = fig.add_subplot(111) #this corresponds to a 3x1 grid, position 1 (top)
for i in range(n_runs):
	ax1.semilogy(a[i],marker='o',linewidth=0,color='none',markeredgecolor='silver', label="Recovered" if i == 0 else "")
ax1.semilogy(x_true,marker='*',linewidth=0,color='blue',label='True')
l = np.arange(0,n_params)
x = ['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15','p16','p17']
ax1.set_xticks(l)
ax1.set_xticklabels(x)

p.legend(loc='best')

p.tight_layout()
p.show()
