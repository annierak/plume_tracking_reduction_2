from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
from matplotlib.gridspec import GridSpec
from itertools import product

plume_plot_name = 'logistic_prob_sim_'
cdf_plot_name = 'trap_time_course_logistic_prob_sim_'


Ks = -1*np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
x_0s = np.arange(50,601,50)
# x_0s = np.arange(50,151,50)
param_pairs = list(product(x_0s, Ks))
plume_plot_filenames = []
cdf_plot_filenames = []
for (x_0,K) in param_pairs:
    plume_plot_filenames.append(plume_plot_name+'x_0_'+str(x_0)+'_K_'+str(K)+'.png')
    cdf_plot_filenames.append(cdf_plot_name+'x_0_'+str(x_0)+'_K_'+str(K)+'.png')

plume_plot_dct = dict(zip(param_pairs,plume_plot_filenames))
cdf_plot_dct = dict(zip(param_pairs,cdf_plot_filenames))

last_pair = (300,-0.4)
K_increment = -0.1
x_0_increment = 50

def press(event):
    global last_pair,image
    print('press arrow keys', event.key)
    sys.stdout.flush()
    if event.key == 'up':
        new_pair = (last_pair[0],np.round(last_pair[1]+K_increment,1))

    if event.key == 'down':
        new_pair = (last_pair[0],np.round(last_pair[1]-K_increment,1))

    if event.key == 'left':
        new_pair = (last_pair[0]-x_0_increment,last_pair[1])

    if event.key == 'right':
        new_pair = (last_pair[0]+x_0_increment,last_pair[1])

    try:
        plume_plot=matplotlib.image.imread(plume_plot_dct[new_pair])
        cdf_plot=matplotlib.image.imread(cdf_plot_dct[new_pair])
        image1.set_data(plume_plot)
        image2.set_data(cdf_plot)
        fig1.canvas.draw()
        fig2.canvas.draw()
        last_pair = new_pair
    except(KeyError):
        print('edge of parameter set, go other way')


fig_width = 10

fig1=plt.figure(figsize=(fig_width,fig_width),dpi=100)
# gs = GridSpec(2,2)

fig1.canvas.mpl_connect('key_press_event', press)

# plt.ion()

# ax.plot(np.random.rand(12), np.random.rand(12), 'go')
plume_plot=matplotlib.image.imread(plume_plot_dct[last_pair])
cdf_plot=matplotlib.image.imread(cdf_plot_dct[last_pair])

# ax1 = fig.add_subplot(gs[:,0])
image1 = plt.imshow(plume_plot,interpolation='none')
plt.axis('off')

fig2=plt.figure(figsize=(fig_width,fig_width),dpi=100)
fig2.canvas.mpl_connect('key_press_event', press)
fig2.tight_layout()

image2 = plt.imshow(cdf_plot,interpolation='none')
plt.axis('off')




# ax2 = fig.add_subplot(gs[:,1])
# ax2.imshow(cdf_plot)
#
# plt.axis('off')


plt.show()
