# officials modules
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
# my modules
from pathintegralanalytics.pathintegralanalytics import markov_chain_fun as mcf
from pathintegralanalytics.pathintegralanalytics import pathIntegralObjects as pio
from pathintegralanalytics.pathintegralanalytics import LatticePedSimulation as sim
import new_plot_functions as plf
import trainf10_operativefile as op
# Setup
os.system('clear')
verbose = False
par = {'index_sort': 'unixepoch'
       , 'DataInterface_type': 'prorail_single_file'
       , 'verbose': True

       , 'dtype_Amatrix': np.uint16
       , 'reduce_rows_to': 10000000
       , 'Lx': 23000
       , 'Ly': 10000
       , 'Dx': 200
       , 'Dy': 100
       # , 'bins_hist_vtk': [
       #      np.linspace(0, 230, 200),
       #      np.linspace(-1, 1, 100),
       #      np.linspace(0, 200, 200)
       #      ]
       , 'delta_pedestrian_state_dimension': 1

       , 'window_length_loc': 31
       , 'polyorder_loc': 7
       , 'cut_short_trj': (100, 250)

       , 'calc_velocity': mcf.calc_velocity_SG
       , 'scale': 'D'

       , 'grb': 'pid'

       , 'experiment': 'trainf10'
       , 'datatype': 'RealData'
       , 'format': '.pdf'
       , 'crop_floorfield': False  # only executed if True
       , 'floorfield_dim': {'x_origin': 0
                            , 'y_origin': 0
                            , 'width': 190
                            , 'height': 200}  # used in prorail classes and histogram creation
       , 'use_improved_grid_calculation': True  # this replaces the use of Lx and Ly
       , 'drop_invalid_transitions': False  # don't think this is used anymore
       , 'normalization_threshold': 10 ** -9  # used to assess whether transition matrix is correctly normalized
       , 'normalization_threshold_exponent': 7
       , 'rotate_transition_matrices': False  # this is for the move and norm_move matrices
       }
source_file_path, target_file_path = op.ask_path(
    default_devel_path='/Users/dcm/analisi2022tesi_master/datasets/FF10_data10_SP_PidNum_20_OnePid_PidNum_20_processed.csv'
    , default_entire_path='/Users/dcm/analisi2022tesi_master/datasets/FF10_data10_AllMaster_processed.csv')
# A: set default values for following variables
MyNumPid_max = 2000
MyDistance_min = 200
MyDistance_max = 500
MyRstep_min = 150
MyRstep_max = 300
# A: ask for different values
MyNumPid_max = op.choose_var_default(MyNumPid_max, namevar='MyNumPid_max')
MyDistance_min = op.choose_var_default(MyDistance_min, namevar='MyDistance_min')
MyDistance_max = op.choose_var_default(MyDistance_max, namevar='MyDistance_max')
MyRstep_min = op.choose_var_default(MyRstep_min, namevar='MyRstep_min')
MyRstep_max = op.choose_var_default(MyRstep_max, namevar='MyRstep_max')
# B: read the csv file, make the list of pids and number of pids
df = pd.read_csv(source_file_path)
print(df.keys())
Pid_list, Num_pid = op.make_pid_list_and_count(df)
Num_pid_iniziali = Num_pid
# D: Drop pids from dataframe by DISTANCE value
dropbyDist = input('[dropbyDist] Return for Yes, or write n/N : ')
if dropbyDist == 'n' or dropbyDist == 'N' or dropbyDist == 'No' or dropbyDist == 'NO':
    print('No drop by pid distance')
else:
    print('Go drop by pid distance')
    df = op.drop_by_distance(df, MyDistance_min, MyDistance_max)
print(df.keys())
# B: make the list of pids and number of pids
Pid_list, Num_pid = op.make_pid_list_and_count(df)
# C: Drop pids from dataframe by RSTEP value
dropbypidRstep = input('[dropbypidRstep] Return for Yes, or write n/N : ')
if dropbypidRstep == 'n' or dropbypidRstep == 'N' or dropbypidRstep == 'No' or dropbypidRstep == 'NO':
    print('No drop by pid rstep')
else:
    print('Go drop by pid rstep')
    df, Rstep_min, Rstep_max = op.drop_by_PidRstep_GetMinMax(df, MyRstep_min, MyRstep_max)
print(df.keys())
# B: make the list of pids and number of pids
Pid_list, Num_pid = op.make_pid_list_and_count(df)
# E: Drop pids from dataframe by MAX NUMBER of PIDS value
dropbyNumPid = input('[dropbyNumPid] Return for Yes, or write n/N : ')
if dropbyNumPid == 'n' or dropbyNumPid == 'N' or dropbyNumPid == 'No' or dropbyNumPid == 'NO':
    print('No drop by number of pids')
else:
    print('Go drop by number of pids')
    df = op.drop_by_NumPid(df, MyNumPid_max)
# B: make the list of pids and number of pids
Pid_list, Num_pid = op.make_pid_list_and_count(df)
# Print the firsts rows of the dataframe
print(df[:3])
# G: Save the CSV file
target_file_path = op.save_csv(df, target_file_path, processed=False, reduced=True)
# F: calc dataframe INFOs
dict_info = op.calc_df_info(df, Num_pid_iniziali=Num_pid_iniziali)
# H: Save and print the TXT FILE
cosa_scrivere = op.save_txt_info(dict_info, target_file_path)
print(cosa_scrivere)

# Technical PIP-STOP
proceed = input('\n\n IT S TIME TO CHECK \nTo proceed press RETURN, otherwise CTRL-C\n')

# I: generate instance PedDataIface and target PROCESSED file path then print info
pedDataIface, proc_target_file_path = op.get_PDIface(target_file_path, par, rename_col=True)
print(proc_target_file_path)
print(pedDataIface.df.keys())
# J: calculate transition matrix for D2Q9 and save new dataframe
dict_transD2Q9 = op.calc_transD2Q9(pedDataIface, par)
new_df = dict_transD2Q9['return_tracks']
# G: Save the CSV file of the new dataframe
proc_target_file_path = op.save_csv(new_df, proc_target_file_path, processed=True, reduced=False)
# K: Plot with PLF the D2Q9 maps in a 3x3 matrix figure
#plf.make_D2Q9_matrix_heatmap(dict_transD2Q9['norm_move'], filename='figure_trainf10_')
# L: Plot with PLF three figures: lines, heatmap and velocities from first df
#plf.just_plot_three(target_file_path, par, add_info="preTrans")
# L: Plot with PLF three figures: lines, heatmap and velocities from second df
#plf.just_plot_three(proc_target_file_path, par, add_info="postTrans")
"""
# H: Save and print the TXT FILE new df
cosa_scrivere = op.save_txt_info_generic_df(target_file_path)
print(cosa_scrivere)
"""

print('\n  --- SIMULATION ZONE ---  \n')
# Technical PIP-STOP
proceed = input('\n\n IT S TIME TO CHECK \nTo proceed press RETURN, otherwise CTRL-C\n')

print(dict_transD2Q9['data_type'])
print(dict_transD2Q9['move'].shape)
print(dict_transD2Q9.keys())
print(dict_transD2Q9['return_tracks'][:3])

print('\n  ---  \n')

par.update({'simulated_steps': 2})
par.update({'num_pid': 3})
par.update({'starting_position': mcf.rand_initial_position_correlation_XY_D2Q9})

print('\n  ---  \n')
dict_simD2Q9 = pio.SpaceTime_transitions_D2Q9(pedDataIface, par)
dict_simD2Q9.create_transition_matrix()

print('\n  -1-  \n')
print(dict_simD2Q9.results['data_type'])
print(dict_simD2Q9.results.keys())
print(dict_simD2Q9.results['return_tracks'][:3])
print(dict_simD2Q9.results['move'].shape)

print('\n -2- \n')
simD2Q9 = sim.LatticeSimulation(dict_simD2Q9, par)
simD2Q9.initialize_positions()
simD2Q9.step_forward(par['simulated_steps'])
df_simD2Q9 = simD2Q9.make_df_from_position_list()


""" ERROR
Traceback (most recent call last):
  File "/Users/dcm/analisi2022tesi_master/trainf10_from_CSV_to_D2Q9_2_sim.py", line 167, in <module>
    simD2Q9.step_forward(par['simulated_steps'])
  File "/Users/dcm/analisi2022tesi_master/pathintegralanalytics/pathintegralanalytics/LatticePedSimulation.py", line 93, in step_forward
    current_step = self.step(previous_step, current_probability_distrib, loc_time + 1
  File "/Users/dcm/analisi2022tesi_master/pathintegralanalytics/pathintegralanalytics/LatticePedSimulation.py", line 113, in step
    return self._simulation_protocols.step(*args)
  File "/Users/dcm/analisi2022tesi_master/pathintegralanalytics/pathintegralanalytics/LatticePedSimulation.py", line 127, in step
    return mcf.passo(*args)
  File "/Users/dcm/analisi2022tesi_master/pathintegralanalytics/pathintegralanalytics/markov_chain_fun.py", line 1784, in passo
    rand_k = np.random.choice(vet_k, p=n)
  File "mtrand.pyx", line 939, in numpy.random.mtrand.RandomState.choice
ValueError: probabilities do not sum to 1
"""



"""
target_file_sim = '/Users/dcm/analisi2022tesi_master/simulationDatasets/sim_D2Q9.csv'
target_file_sim = op.save_csv(df_simD2Q9, target_file_sim)
df = pd.read_csv(target_file_sim)
lines = plf.PositionLines(df, par)
lines.figure_save()
heatmap = PositionHeatmap(df, par)
heatmap.figure_save()
"""



print('\n   --- END ---   \n')

