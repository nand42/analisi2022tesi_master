import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt

from pathintegralanalytics.pathintegralanalytics import markov_chain_fun as mcf
from pathintegralanalytics.pathintegralanalytics import pathIntegralObjects as pio
from pathintegralanalytics.pathintegralanalytics import LatticePedSimulation as sim
from pathintegralanalytics.pathintegralanalytics import new_plot_functions as plf
import trainf10_operativefile as op

os.system('clear')

par = {
    'index_sort': 'unixepoch'
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
    , 'floorfield_dim': {'x_origin': 0,
                         'y_origin': 0,
                         'width': 190,
                         'height': 200}  # used in prorail classes and histogram creation
    , 'use_improved_grid_calculation': True  # this replaces the use of Lx and Ly
    , 'drop_invalid_transitions': False  # don't think this is used anymore
    , 'normalization_threshold': 10 ** -9  # used to assess whether transition matrix is correctly normalized
    , 'normalization_threshold_exponent': 7
    , 'rotate_transition_matrices': False  # this is for the move and norm_move matrices
}

source_file_path, target_file_path = op.ask_path(
    default_devel_path='/Users/dcm/analisi2022tesi_master/datasets/FF10_data10_SP_PidNum_20_OnePid_PidNum_20_processed.csv'
    , default_entire_path='/Users/dcm/analisi2022tesi_master/datasets/FF10_data10_AllMaster_processed.csv')

verbose = True

MyNumPid_max = 2000
MyDistance_min = 200
MyDistance_max = 500
MyRstep_min = 150
MyRstep_max = 300

MyNumPid_max = op.choose_var_default(MyNumPid_max, namevar='MyNumPid_max')
MyDistance_min = op.choose_var_default(MyDistance_min, namevar='MyDistance_min')
MyDistance_max = op.choose_var_default(MyDistance_max, namevar='MyDistance_max')
MyRstep_min = op.choose_var_default(MyRstep_min, namevar='MyRstep_min')
MyRstep_max = op.choose_var_default(MyRstep_max, namevar='MyRstep_max')

print('\n   ---   \n')

df = pd.read_csv(source_file_path)
print(df.keys())

print('\n   ---   \n')

Pid_list, Num_pid = op.make_pid_list_and_count(df)

Num_pid_iniziali = Num_pid

print('\n   ---   \n')

df, Rstep_min, Rstep_max = op.drop_by_PidRstep_GetMinMax(df, MyRstep_min, MyRstep_max)
print(df.keys())

print('\n   ---   \n')

Pid_list, Num_pid = op.make_pid_list_and_count(df)

print('\n   ---   \n')

df = op.drop_by_distance(df, MyDistance_min, MyDistance_max)
print(df.keys())

print('\n   ---   \n')

# df = op.drop_by_NumPid(df, MyNumPid_max)

print('\n   ---   \n')

print(df[:3])

print('\n   ---   \n')

dict_info = op.calc_df_info(df, Num_pid_iniziali=Num_pid_iniziali)
print(dict_info)

print('\n   ---   \n')

target_file_name = op.save_csv(df, target_file_path, processed=True, reduced=True)

print('\n   ---   \n')

cosa_scrivere = op.save_txt_info(dict_info, target_file_name)

print('\n   ---   \n')

print(cosa_scrivere)

print('\n   ---   \n')

