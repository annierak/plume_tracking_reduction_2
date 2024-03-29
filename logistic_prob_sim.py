'''
Simulation where, by varying as input, only
the two parameters controlling the
{distance-successful recovery relationship} (specifically, x_0 and K)
'''

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
    try:
        file_name = 'logistic_prob_sim_x_0_'+str(x_0)+'_K_'+str(K)
        output_file = file_name+'.pkl'


        dt = 0.25
        frame_rate = 20
        times_real_time = 20 # seconds of simulation / sec in video
        capture_interval = int(scipy.ceil(times_real_time*(1./frame_rate)/dt))

        simulation_time = 50.*60. #seconds
        release_delay = 0.*60#/(wind_mag)

        t_start = 0.0
        t = 0. - release_delay


        wind_angle = 7*scipy.pi/8.
        wind_mag = 1.
        # wind_angle = 7*scipy.pi/4.
        wind_param = {
                    'speed': wind_mag,
                    'angle': wind_angle,
                    'evolving': False,
                    'wind_dt': None,
                    'dt': dt
                    }
        wind_field = wind_models.WindField(param=wind_param)

        #traps
        number_sources = 8
        radius_sources = 1000.0
        trap_radius = 0.5
        location_list, strength_list = utility.create_circle_of_sources(number_sources,
                        radius_sources,None)
        trap_param = {
                'source_locations' : location_list,
                'source_strengths' : strength_list,
                'epsilon'          : 0.01,
                'trap_radius'      : trap_radius,
                'source_radius'    : radius_sources
        }

        traps = trap_models.TrapModel(trap_param)

        #Wind and plume objects

        #Odor arena
        xlim = (-1500., 1500.)
        ylim = (-1500., 1500.)
        im_extents = xlim[0], xlim[1], ylim[0], ylim[1]

        source_pos = scipy.array([scipy.array(tup) for tup in traps.param['source_locations']])

        # Set up logistic prob plume object

        logisticPlumes = models.LogisticProbPlume(K,x_0,source_pos,wind_angle)
        #To document the plume parameters, save a reference plot of the plume probability curve

        plt.figure()
        inputs = np.linspace(0,1000,1000)
        outputs = logisticPlumes.logistic_1d(inputs)
        plt.plot(inputs,outputs)
        plt.title('Logistic Curve with K: '+str(K)+', x_0: '+str(x_0),color='purple')
        plt.xlim(0,1000.)
        plt.ylim(-0.02,1.)
        plt.xlabel('Distance from Trap (m)')
        plt.ylabel('Trap Arrival Probability')

        plt.savefig(file_name+'.png',format='png')



        # Setup fly swarm
        wind_slippage = (0.,1.)
        # swarm_size=2000
        swarm_size=20000
        use_empirical_release_data = False

        #Grab wind info to determine heading mean
        wind_x,wind_y = wind_mag*scipy.cos(wind_angle),wind_mag*scipy.sin(wind_angle)

        beta = 1.
        release_times = scipy.random.exponential(beta,(swarm_size,))
        kappa = 2.

        heading_data=None

        #Flies also use parameters (for schmitt_trigger, detection probabilities)
        # determined in
        #fly_behavior_sim/near_plume_simulation_sutton.py

        swarm_param = {
                'swarm_size'          : swarm_size,
                'heading_data'        : heading_data,
                'initial_heading'     : scipy.radians(scipy.random.uniform(0.0,360.0,(swarm_size,))),
                'x_start_position'    : scipy.zeros(swarm_size),
                'y_start_position'    : scipy.zeros(swarm_size),
                'flight_speed'        : scipy.full((swarm_size,), 1.5),
                'release_time'        : release_times,
                'release_delay'       : release_delay,
                'cast_interval'       : [1, 3],
                'wind_slippage'       : wind_slippage,
                'odor_thresholds'     : {
                    'lower': 0.0005,
                    'upper': 0.05
                    },
                'schmitt_trigger':False,
                'low_pass_filter_length':3, #seconds
                'dt_plot': capture_interval*dt,
                't_stop':3000.,
                'cast_timeout':20,
                'airspeed_saturation':True
                }


        swarm = swarm_models.ReducedSwarmOfFlies(wind_field,traps,param=swarm_param,
            start_type='fh')

        # xmin,xmax,ymin,ymax = -1000,1000,-1000,1000


        xmin,xmax,ymin,ymax = -1000,1000,-1000,1000

        # plt.show()
        # raw_input()
        while t<simulation_time:
            for k in range(capture_interval):
                #update flies
                print('t: {0:1.2f}'.format(t))
                swarm.update(t,dt,wind_field,logisticPlumes,traps)
                t+= dt
                # time.sleep(0.001)
        with open(output_file, 'w') as f:
            pickle.dump((wind_field,swarm),f)
    except(ValueError):
        print('p>1 error for (x_0,K) pair '+str((x_0,K)))
        sys.exit()

# f(300,-0.4)


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
x_0s = np.arange(350,601,50)

with poolcontext(processes=10) as pool:
    pool.map(f_unpack, product(x_0s, Ks))


# pool.map(main,mags)
