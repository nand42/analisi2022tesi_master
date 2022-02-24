import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from pathintegralanalytics.pathintegralanalytics import markov_chain_fun as mcf
from pathintegralanalytics.pathintegralanalytics import pathIntegralObjects as pio
from pathintegralanalytics.pathintegralanalytics import new_plot_functions as plf
from pathintegralanalytics.pathintegralanalytics import LatticePedSimulation as sim

os.system('clear')

if True:  # print path of some modules
    print('\n\n--- path modulo mcf ---')
    print(mcf.__file__)
    print('---         ---         ---')
    print('--- path modulo pia ---')
    print(pio.__file__)
    print('---         ---         ---')
    print('--- path modulo plf ---')
    print(plf.__file__)
    print('---         ---         ---')
    print('--- path modulo sim ---')
    print(sim.__file__)
    print('---         ---         ---\n\n')


def ask_path(default_devel_path='/Users/dcm/analisi2022tesi_master/datasets/dev_FF10_data10.csv'
             , default_entire_path='/Users/dcm/analisi2022tesi_master/datasets/FF10_data10.csv'):
    develop_file = True

    print('[!!!] - Default running with development CSV? ')
    choose_develop_file = input('Return for Yes, or write n/N : ')
    if choose_develop_file == 'n' or choose_develop_file == 'N' or choose_develop_file == 'No' or choose_develop_file == 'NO':
        develop_file = False

    if develop_file:
        source_file_path = default_devel_path
        target_file_path = source_file_path[:-4] + '_op_.csv'
        print('\n[!!!] - DEVELOPMENT')
    else:
        choose_develop_file = input('Do you want to use the STANDARD PATH? then Return. \nIf you want a NEW PATH say n/N/No : ')
        if choose_develop_file == 'n' or choose_develop_file == 'N' or choose_develop_file == 'No' or choose_develop_file == 'NO':
            source_file_path = input('NEW PATH at: ')
            target_file_path = source_file_path[:-4] + '_op_.csv'
            print('\n[!!!] - NEW PATH')
        else:
            source_file_path = default_entire_path
            target_file_path = source_file_path[:-4] + '_op_.csv'
            print('\n[!!!] - THE WHOLE DATASET trainf10')

    print('\nThe selected SOURCE FILE is: \n >>  ' + source_file_path + '  <<\n')

    return source_file_path, target_file_path


def choose_var_default(var, namevar='VAR'):
    print('\nActual value of ' + str(namevar) + ' = ' + str(var))
    change = input('If you like it type Return. \nOr if you want to change it type something (y/Y): ')
    newvar = var
    if change != '':
        newvar = input('Type the new value [int]: ')
        if newvar == '':
            newvar = var
            print('CONFIRMATION: value not changed.')
        else:
            newvar = int(newvar)
            print('CONFIRMATION: of the new selected value for ' + str(namevar) + ' = ' + str(newvar))
    else:
        newvar = var
        print('CONFIRMATION: value not changed.')
    return newvar


def drop_by_PidRstep_GetMinMax(df, min_PidRstep, max_PidRstep):
    df = df.drop(df[df.Rstep_len < min_PidRstep].index)
    df = df.drop(df[df.Rstep_len > max_PidRstep].index)
    Rstep_min = df.Rstep_len.min()
    Rstep_max = df.Rstep_len.max()
    print('\nSelected pids by the value of Rstep:\nmin value = ' + str(Rstep_min) + '\nmax value = ' + str(Rstep_max))
    return df, Rstep_min, Rstep_max


def calc_distance_and_make_zero_df(df, start=1, end=1):
    df = df.assign(distance=0)
    new_df = df[0:0]
    df_zero = df[0:0]
    print('\nAdded a key distance to the dataframe keys: ')
    print(df.keys())
    PidList, Num_pid = make_pid_list_and_count(df)

    for pid in PidList:
        df_pid = df.query('pid == @pid')

        xi = df_pid.query('Rstep == @start').head(1)['x'].iloc[0]
        yi = df_pid.query('Rstep == @start').head(1)['y'].iloc[0]
        xf = df_pid.query('Rstep == Rstep_len-@end').head(1)['x'].iloc[0]
        yf = df_pid.query('Rstep == Rstep_len-@end').head(1)['y'].iloc[0]

        distance = round(math.sqrt(((xi - xf) ** 2 + (yi - yf) ** 2)))

        df_pid = df_pid.assign(distance=distance)
        new_df = pd.concat([new_df, df_pid])

    print('Added new colum DISTANCE to the dataframe and created a new equal empty dataframe')
    return new_df, df_zero


def make_pid_list_and_count(df):
    Pid_list = list(set(list(df.pid.unique())))
    Num_pid = len(Pid_list)
    print('\nCreated the pid list \n With number uniques pids = ' + str(Num_pid))
    return Pid_list, Num_pid


def drop_by_PidList(df, PidList=[]):
    if PidList == []:
        print('\nCalculating Pid List first')
        PidList, Num_pid = make_pid_list_and_count(df)
    print('\nPid List available')
    new_df = df[0:0]
    for pid in PidList:
        df_pid = df.query('pid == @pid')
        # if verbose: print('(pid) = (' + str(pid) + ')')
        new_df = pd.concat([new_df, df_pid])
    print('Dropped pid list')
    return new_df


def drop_by_distance(df, min_distance, max_distance):
    if 'distance' not in df.index:
        print('\nCalculating DISTANCES first')
        df, df_zero = calc_distance_and_make_zero_df(df)

    df = df.drop(df[df.distance < min_distance].index)
    df = df.drop(df[df.distance > max_distance].index)

    print('Dropped pids by distance')
    return df


def drop_by_NumPid(df, max_NumPid, PidList=[]):
    if PidList == []:
        print('\nCalculating Pid List first')
        PidList, Num_pid = make_pid_list_and_count(df)
    PidList = list(set(PidList))[:max_NumPid]  # takes only the firsts MyNumPid_max PIDs
    new_df = drop_by_PidList(df, PidList)
    print('Dropped pids by Max pid number')
    return new_df


#  ---  ---  ---  Beginning of the program  ---  ---  ---


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

source_file_path, target_file_path = ask_path(
    default_devel_path='/Users/dcm/analisi2022tesi_master/datasets/FF10_data10_SP_PidNum_20_OnePid_PidNum_20_processed.csv'
    , default_entire_path='/Users/dcm/analisi2022tesi_master/datasets/FF10_data10_AllMaster_processed.csv')

verbose = True

MyNumPid_max = 2000
MyDistance_min = 200
MyDistance_max = 500
MyRstep_min = 150
MyRstep_max = 300

MyNumPid_max = choose_var_default(MyNumPid_max, namevar='MyNumPid_max')
MyDistance_min = choose_var_default(MyDistance_min, namevar='MyDistance_min')
MyDistance_max = choose_var_default(MyDistance_max, namevar='MyDistance_max')
MyRstep_min = choose_var_default(MyRstep_min, namevar='MyRstep_min')
MyRstep_max = choose_var_default(MyRstep_max, namevar='MyRstep_max')

print('\n   ---   \n')

df = pd.read_csv(source_file_path)
print(df.keys())

print('\n   ---   \n')

Pid_list, Num_pid = make_pid_list_and_count(df)

print('\n   ---   \n')

df, Rstep_min, Rstep_max = drop_by_PidRstep_GetMinMax(df, MyRstep_min, MyRstep_max)
print(df.keys())

print('\n   ---   \n')

Pid_list, Num_pid = make_pid_list_and_count(df)

print('\n   ---   \n')

df = drop_by_distance(df, MyDistance_min, MyDistance_max)
print(df.keys())

print('\n   ---   \n')

df = drop_by_NumPid(df, MyNumPid_max)

print('\n   ---   \n')

print(df[:3])

print('\n   ---   \n')
print('ok')








