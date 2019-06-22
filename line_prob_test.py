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


def main(x_0,K):#np.arange(0.4,3.8,0.2):

    file_name = 'logistic_prob_sim_x_0_'+str(x_0)+'_K_'+str(K)
    output_file = file_name+'.pkl'


    dt = 0.25
    frame_rate = 20
    times_real_time = 20 # seconds of simulation / sec in video
    capture_interval = int(scipy.ceil(times_real_time*(1./frame_rate)/dt))

    simulation_time =2.*60. #seconds
    release_delay = 0.*60#/(wind_mag)

    t_start = 0.0
    t = 0. - release_delay

    # Set up figure
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111)



    wind_angle = 3*scipy.pi/2
    wind_mag = 1.6
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
    number_sources = 1
    radius_sources = 1000.0
    trap_radius = 0.5

    location_list = [(0,0)]
    strength_list = [1]


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
    xlim = (-500., 500.)
    ylim = (-1500., 0.)
    im_extents = xlim[0], xlim[1], ylim[0], ylim[1]

    source_pos = scipy.array([scipy.array(tup) for tup in traps.param['source_locations']])

    # Set up logistic prob plume object

    logisticPlumes = models.LogisticProbPlume(K,x_0,source_pos,wind_angle)
    #To document the plume parameters, save a reference plot of the plume probability curve

    #Setup fly swarm
    wind_slippage = (0.,0.)
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
            'initial_heading'     : np.pi*np.ones(swarm_size),
            'x_start_position'    : 100*scipy.ones(swarm_size),
            'y_start_position'    : np.linspace(0,-1000,swarm_size),
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

    #Concentration plotting
    conc_d = logisticPlumes.conc_im(im_extents)

    cmap = matplotlib.colors.ListedColormap(['white', 'orange'])
    cmap = 'YlOrBr'

    conc_im = plt.imshow(conc_d,extent=im_extents,
        interpolation='none',cmap = cmap,origin='lower')

    plt.colorbar()


    xmin,xmax,ymin,ymax = (-500., 500.,-1500., 0.)

    buffr = 100
    ax.set_xlim((xmin-buffr,xmax+buffr))
    ax.set_ylim((ymin-buffr,ymax+buffr))

    #Initial fly plotting
    #Sub-dictionary for color codes for the fly modes
    Mode_StartMode = 0
    Mode_FlyUpWind = 1
    Mode_CastForOdor = 2
    Mode_Trapped = 3


    edgecolor_dict = {Mode_StartMode : 'blue',
    Mode_FlyUpWind : 'red',
    Mode_CastForOdor : 'red',
    Mode_Trapped :   'black'}

    facecolor_dict = {Mode_StartMode : 'blue',
    Mode_FlyUpWind : 'red',
    Mode_CastForOdor : 'white',
    Mode_Trapped :   'black'}


    fly_edgecolors = [edgecolor_dict[mode] for mode in swarm.mode]
    fly_facecolors =  [facecolor_dict[mode] for mode in swarm.mode]
    fly_dots = plt.scatter(swarm.x_position, swarm.y_position,
        edgecolor=fly_edgecolors,facecolor = fly_facecolors,alpha=0.9)

    #Put the time in the corner
    (xmin,xmax) = ax.get_xlim();(ymin,ymax) = ax.get_ylim()
    text = '0 min 0 sec'
    timer= ax.text(xmax,ymax,text,color='r',horizontalalignment='right')
    ax.text(1.,1.02,'time since release:',color='r',transform=ax.transAxes,
        horizontalalignment='right')



    # #traps
    for x,y in traps.param['source_locations']:

        #Black x
        plt.scatter(x,y,marker='x',s=50,c='k')

        # Red circles
        # p = matplotlib.patches.Circle((x, y), 15,color='red')
        # ax.add_patch(p)

    #Remove plot edges and add scale bar
    fig.patch.set_facecolor('white')
    plt.plot([-900,-800],[900,900],color='k')#,transform=ax.transData,color='k')
    ax.text(-900,820,'100 m')
    plt.axis('off')

    #Fly behavior color legend
    for mode,fly_facecolor,fly_edgecolor,a in zip(
        ['Dispersing','Surging','Casting','Trapped'],
        facecolor_dict.values(),
        edgecolor_dict.values(),
        [0,50,100,150]):

        plt.scatter([1000],[-600-a],edgecolor=fly_edgecolor,
        facecolor=fly_facecolor, s=20)
        plt.text(1050,-600-a,mode,verticalalignment='center')


    plt.ion()


    # plt.show()
    # raw_input()
    while t<simulation_time:
        for k in range(capture_interval):
            #update flies
            print('t: {0:1.2f}'.format(t))
            swarm.update(t,dt,wind_field,logisticPlumes,traps)
            t+= dt
            plt.pause(0.001)
            # time.sleep(0.001)
        release_delay = release_delay/60.
        text ='{0} min {1} sec'.format(
            int(scipy.floor(abs(t/60.))),int(scipy.floor(abs(t)%60.)))
        timer.set_text(text)
        #
        # '''plot the flies'''
        fly_dots.set_offsets(scipy.c_[swarm.x_position,swarm.y_position])

        fly_edgecolors = [edgecolor_dict[mode] for mode in swarm.mode]
        fly_facecolors =  [facecolor_dict[mode] for mode in swarm.mode]
        #
        fly_dots.set_edgecolor(fly_edgecolors)
        fly_dots.set_facecolor(fly_facecolors)



    #the expected curve
    plt.figure(2)
    inputs = np.linspace(0,1000,1000)
    outputs = logisticPlumes.logistic_1d(inputs)
    plt.plot(inputs,outputs,'r')
    plt.title('Logistic Curve with K: '+str(K)+', x_0: '+str(x_0),color='purple')
    plt.xlim(0,1000.)
    plt.ylim(-0.02,1.)
    plt.xlabel('Distance from Trap (m)')
    plt.ylabel('Trap Arrival Probability')

    #Examine the empirical success rate by distance
    passed_thru_mask = (swarm.mode == Mode_StartMode)
    num_bins = 100;max_trap_distance = 1000
    bin_width = max_trap_distance/num_bins
    fly_release_density = swarm_size/max_trap_distance
    fly_release_density_per_bin = fly_release_density*bin_width

    bins=np.linspace(0,max_trap_distance,num_bins)
    n_successes,_ = np.histogram(-1.*swarm.y_position[passed_thru_mask],bins)
    n_successes = n_successes.astype(float)
    # print(n_successes)
    # print(fly_release_density_per_bin)
    plt.plot(bins[:-1],1.-(n_successes/(fly_release_density_per_bin)),'o  ')
    plt.show()
    raw_input()


    with open(output_file, 'w') as f:
        pickle.dump((wind_field,swarm),f)

main(300,-0.4)



# pool.map(main,mags)
