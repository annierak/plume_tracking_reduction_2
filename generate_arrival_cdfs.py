
import time
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import matplotlib
import matplotlib.patches
matplotlib.use("Agg")
import sys
import itertools
import h5py
import json
import cPickle as pickle
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np

import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.wind_models as wind_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt
from pompy import models,processors

def f(x_0,K):

    wind_angle = 7*scipy.pi/8.
    wind_mag = 1.

    file_name = 'logistic_prob_sim_x_0_'+str(x_0)+'_K_'+str(K)
    output_file = file_name+'.pkl'


    with open(output_file, 'r') as f:
        (_,swarm) = pickle.load(f)


    #Trap arrival plot

    num_bins = 120

    trap_num_list = swarm.get_trap_nums()


    peak_counts = scipy.zeros(len(trap_num_list))
    peak_counts = scipy.zeros(8)
    rasters = []

    fig = plt.figure(figsize=(7, 11))

    fig.patch.set_facecolor('white')

    labels = ['N','NE','E','SE','S','SW','W','NW']

    sim_reorder = scipy.array([3,2,1,8,7,6,5,4])

    # plt.ion()

    #Simulated histogram
    # for i in range(len(trap_num_list)):
    axes = []
    for i in range(8):

        row = sim_reorder[i]-1
        # ax = plt.subplot2grid((len(trap_num_list),1),(i,0))
        ax = plt.subplot2grid((8,1),(row,0))
        t_sim = swarm.get_time_trapped(i)
        # print(t_sim)
        # raw_input()

        if len(t_sim)==0:
            ax.set_xticks([0,10,20,30,40,50])
            trap_total = 0
            pass
        else:
            t_sim = t_sim/60.
            (n, bins, patches) = ax.hist(t_sim,num_bins,cumulative=True,
            histtype='step',
            range=(0,max(t_sim)))

            # n = n/num_iterations
            # trap_total = int(sum(n))
            # trap_total = int(n[-1])
            try:
                peak_counts[i]=max(n)
            except(IndexError):
                peak_counts[i]=0


        if sim_reorder[i]-1==0:
            ax.set_title('Cumulative Trap Arrivals \n  K: '+str(K)+', x_0: '+str(x_0))

        ax.set_xlim([0,50])
        # plt.pause(0.001)
        # ax.set_yticks([ax.get_yticks()[0],ax.get_yticks()[-1]])
        plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True)
        # ax.text(-0.1,0.5,str(trap_total),transform=ax.transAxes,fontsize=20,horizontalalignment='center')
        # ax.text(-0.01,1,trap_total,transform=ax.transAxes,fontsize=10,
        #     horizontalalignment='center',verticalalignment='center')
        ax.text(-0.1,0.5,str(labels[sim_reorder[i]-1]),transform=ax.transAxes,fontsize=20,
            horizontalalignment='center',verticalalignment='center')
        if sim_reorder[i]-1==7:
            ax.set_xlabel('Time (min)',x=0.5,horizontalalignment='center',fontsize=20)
            plt.tick_params(axis='both', which='major', labelsize=15)
        else:
            ax.set_xticklabels('')
        axes.append(ax)

    for ax in axes:
        # row = sim_reorder[i]-1
        # ax = plt.subplot2grid((8,1),(row,0))
        # ax.set_ylim([0,max(peak_counts)])
        ax.set_ylim([0,400])
        ax.set_yticks([ax.get_yticks()[0],ax.get_yticks()[-1]])
        # raw_input()



    # plt.show()

    png_file_name = 'trap_time_course_'+file_name
    plt.savefig(png_file_name+'.png',format='png',bbox_inches='tight')

import multiprocessing
from itertools import product
from contextlib import contextmanager

def f_unpack(args):
    return f(*args)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

Ks = -1*np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
# x_0s = np.arange(50,601,50)
# x_0s = np.arange(50,301,50)
x_0s = np.arange(350,601,50)



# f(300,-0.4)
#
with poolcontext(processes=10) as pool:
    pool.map(f_unpack, product(x_0s, Ks))
