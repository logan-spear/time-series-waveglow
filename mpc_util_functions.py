# %matplotlib inline
import datetime
import pandas as pd
import matplotlib
import numpy as np
import cvxpy as cvx
from matplotlib import pyplot as plt

# import local copy of cvxpower
import os
import sys
# sys.path.insert(0, os.path.abspath('../'))
from cvxpower.cvxpower import *

# matplotlib.rc("figure", figsize=(16,6))
matplotlib.rc("lines", linewidth=2)
#matplotlib.rc("font", serif='Computer Modern Roman')
# matplotlib.rcParams['mathtext.fontset'] = 'cm'
# matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# matplotlib.rc('text', usetex=True)

graphdir = './'
figsize=(8,5)


########################################################################################



baseline = pd.read_pickle('baseline_forecaster_params/wind_baseline.pickle')
autoreg_residual_params = pd.read_pickle('baseline_forecaster_params/residual_params.pickle')
sigma_residual_errors = pd.read_pickle('baseline_forecaster_params/sigma_epsilon.pickle')
train = pd.read_pickle('baseline_forecaster_params/wind_power_train.pickle')
test = pd.read_pickle('baseline_forecaster_params/wind_power_test.pickle')
p_wind = pd.concat([train,test])
train_residuals = pd.read_pickle('baseline_forecaster_params/residual_samples.pickle')
# del train
# del test


########################################################################################

# CONSTANTS
# Use a 10:1 ratio for wind power to storage power and a 1:1 ratio for storage capacity (hours) to power
wind_power_max = 16.  
wind_power_min = 0.  

# Realistic numbers for battery (makes for uninteresting optimization problems)
num_hours_battery = 4 # number of hours takes for battery to full charge/discharge
storage_capacity = wind_power_max / 10 # MWh, using a 10:1 ratio with wind power
battery_charge_max = storage_capacity / num_hours_battery
battery_discharge_max = storage_capacity / num_hours_battery

# Going back to original paper numbers, problem isn't interesting when battery is so small
# storage_capacity = 100 # MWh
# battery_charge_max = storage_capacity / 10. # a full charge in 10 hours
# battery_discharge_max = storage_capacity / 10. # a full discharge in 10 hours

initial_storage = storage_capacity // 2

gas_power_max = 12  #MW
gas_power_min = 0 #MW
gas_alpha = .5
gas_beta = 40.

# len of each time interval, in hours
len_interval = (baseline.index[1] - baseline.index[0]).seconds/3600
intervals_per_day = int(24 / len_interval)

# analysis horizon
T = intervals_per_day * 31

# we test on the first off-sample month
sim_start_time = 70080 # Timestamp('2012-01-01 00:00:00')
sim_end_time = sim_start_time + T

assert len(baseline) > sim_end_time
assert baseline.index[sim_start_time].year >= 2012 # out of sample

MPC_final_energy_price = 80 / len_interval


#wind_power_available = test[target_output.index]

target_output = pd.Series(data= 8., #baseline[sim_start_time:sim_end_time].mean(),
                          index=baseline[sim_start_time:sim_start_time + 2*T].index)

#p_wind[sim_start_time:sim_end_time].plot(figsize=(12,3), label='Wind power')
# plt.figure(figsize=(16,6))
# plt.subplot2grid((4,1), (0,0), rowspan=3)
# y = p_wind[sim_start_time:sim_end_time]
# plt.plot(y.index, y.values, label='Wind power')
# target_output[:T].plot(label='Target output')
# plt.gca().set_xlim(['01-01-2012', '01-31-2012'])
# plt.ylabel('power (MW)')
# plt.xlabel('time')

# plt.legend(loc='lower left')
# plt.savefig('wind_data.pdf')

########################################################################################



def cost_per_unit(power):
    return (gas_alpha * power**2 + gas_beta * power)/power

########################################################################################

def make_network(T, K):
    """Define network and parameters for data."""
    target_output = Parameter((T,K))
    wind_power_available = Parameter((T,K))
    initial_storage = Parameter((1,1))
    final_energy_price = Parameter((1,1))
    final_storage = Parameter((1,1))
    out = FixedLoad(power = target_output, name="Target output")
    wind_gen = Generator(alpha=0, beta=0, power_min=0, 
                         power_max=wind_power_available, name="Wind generator")
    gas_gen = Generator(alpha=gas_alpha, beta=gas_beta, 
                        power_min=gas_power_min, power_max=gas_power_max, name="Gas generator",
                       len_interval = len_interval)
    storage = Storage(discharge_max=battery_discharge_max, charge_max=battery_charge_max, 
                      energy_max=storage_capacity, 
                      energy_init = initial_storage, 
                      energy_final = final_storage,
                     len_interval = len_interval,
                     final_energy_price = final_energy_price)
    net = Net([wind_gen.terminals[0],
               gas_gen.terminals[0],
               storage.terminals[0],
               out.terminals[0]])
    my_network = Group([wind_gen, gas_gen, storage, out], [net])
    my_network.init_problem(time_horizon=T, num_scenarios = K)
    return target_output, wind_power_available, initial_storage, final_storage, final_energy_price, my_network

########################################################################################

def print_and_plot_stats(wind_power_avail, wind_power_used, gas_power, output, cost, plot_stuff=False):
    assert len(output) == len(wind_power_avail)
    assert len(output) == len(wind_power_used)
    assert len(output) == len(gas_power)

    if plot_stuff:
        plt.figure(figsize=(9,3.5))
        plt.subplot2grid((10,1), (0,0), rowspan=9)
        plt.plot(wind_power_avail.index, wind_power_avail.values, label='avail.', alpha=.8)
        pd.Series(data = wind_power_used, 
                  index = wind_power_avail.index).plot(label='used', alpha=.8)
        #output.plot(label='target', style='--', color='k', alpha=.8)
        plt.legend(loc='lower left')
        plt.gca().set_xlim(['01-01-2012', '01-31-2012'])
        plt.ylabel('Power (MW)')
        plt.xlabel('time')
    #plt.gcf().autofmt_xdate()
    # plt.savefig('wind_curtailment.pdf')

    total_output = sum(output)/len(output) 
    total_wind_power_avail = sum(wind_power_avail)/len(output)
    total_gas_gen = sum(gas_power)/len(output)

    print('(Values are daily averages.)\n')
    print('Energy sold:\t\t%.2f MWh\nWind energy avail.:\t%.2f MWh\nGas gener. output:\t%.2f MWh\nWind energy used:\t%.2f MWh\nWind energy lost:\t%.2f MWh' % (
        24*total_output,
        24*total_wind_power_avail,
        24*total_gas_gen,
        24*np.mean(wind_power_used),
        24*(total_wind_power_avail - np.mean(wind_power_used))
    ))

    print('\nEnergy sold (at $80/MWh):    $%.2f' % (24 * total_output *80))
    print('Cost of gas generated energy:  $%.2f' % (24 * cost / (len(output))))


########################################################################################

def plot(results, target_output, energy_stored, methodname):
    ax = results.plot(figsize=(9,3.5), index=target_output.index)
    plt.subplots_adjust(hspace=0.075)
    ax[0].set_ylabel('power (MW)')
    ax[0].set_xlim(['01-01-2012', '01-31-2012'])
    ax[0].legend(['gen.', 'batt.', 'target', 'wind'], loc='lower left')
    ax[0].tick_params(axis="x", which="both", bottom="off", top="off",
                  labelbottom="off", left="off", right="off", labelleft="off")
    ax[1].clear() 
    ax[1].plot(target_output.index, energy_stored)
    ax[1].set_ylabel('Energy (MWh)')
    ax[1].set_xlim(['01-01-2012', '01-31-2012'])
    ax[1].set_ylim([0, 50])
    plt.xlabel('time')
    plt.savefig(graphdir+'wind_%s_results.pdf'%methodname)
    
    prices = list(results.price.values())[0].flatten()
    plt.figure(figsize=(9,3.5))
    plt.plot(target_output.index, prices)
    plt.ylabel('net price (\$)')
    
########################################################################################

def predict_wind(p_wind, baseline, autoreg_residual_params, t, M, L, K = 1):
    past = p_wind[t-M:t]
    past_baseline = baseline[t-M:t]
    fut_baseline = baseline[t:t+L]
    pred = list(reversed(past-past_baseline)) @ autoreg_residual_params
    pred = pd.Series(pred, index=fut_baseline.index)
    pred += fut_baseline
    pred = np.maximum(wind_power_min, pred)
    pred = np.minimum(wind_power_max, pred)
    return pred


########################################################################################

def run_net_mpc(net, K):
    '''
    Run MPC using scenarios generated from the given WaveGlow model in net
    K is the number of scenarios to use
    '''
    
    T_MPC = intervals_per_day
#     K = 20 # Number of scenarios to sample

    energy_stored = np.empty(T)
    target_output_MPC, wind_power_available_MPC, initial_storage_MPC, final_storage_MPC, final_energy_price, MPC_network = \
        make_network(T_MPC, K)

    initial_storage_MPC.value = np.matrix(initial_storage)
    final_storage_MPC.value = np.matrix(initial_storage)
    final_energy_price.value = np.matrix(MPC_final_energy_price)


    def make_forecasts(t):
        '''
        Populates the cvx parameters target_output.MPC and wind_power_available_MPC
        target_output_MPC is always the same
        wind_power_available_MPC.value is a matrix of shape (T,K), where T is the length of one forecast (96)
        and K is the number of scenarios (specified outside of this function). Each column represents
        one forecast scenario. The first scalar element of every forecast is p_wind[sim_start_time+t]
        '''
        target_output_MPC.value = np.tile(target_output[t:t+T_MPC], (K,1)).T

        #### Logan's code ####
        # First, create the context torch tensors to be used as context for the model
        context = p_wind[(sim_start_time+t+1-T_MPC):(sim_start_time+t+1)].values
        context = np.reshape(context, (1, -1, 1))
        context = np.repeat(context, K, axis=0)
        context = torch.FloatTensor(context)

        # Now, generate forecasts using this context:
        scenarios = net.generate(context).numpy()

        # Take only the first 95 entries, make the first entry p_wind[sim_start_time+t] for all
        scenarios = np.hstack((
            np.matrix([p_wind[sim_start_time+t]]*K).T,
            scenarios[:, :-1]
        ))

        # Max and min out the wind power
        scenarios = np.maximum(wind_power_min, scenarios)
        scenarios = np.minimum(wind_power_max, scenarios)
        wind_power_available_MPC.value = scenarios.T


    def implement(t):
        energy_stored[t] = MPC_network.devices[2].energy.value[0,0]
        initial_storage_MPC.value = np.matrix(energy_stored[t])

        
    print("Results for K = ", K)
    cost_MPC, MPC_results = \
        run_mpc(MPC_network, T, make_forecasts, implement, verbose = False, solver='ECOS')

    wind_gen, gas_gen, storage = MPC_network.devices[0:3]

    print_and_plot_stats(wind_power_avail =  p_wind[sim_start_time:sim_end_time], 
                         wind_power_used = -MPC_results.power[(wind_gen, 0)].flatten(), 
                         gas_power = -MPC_results.power[(gas_gen,0)].flatten(), 
                         output =  target_output[:T],
                         cost = cost_MPC)



