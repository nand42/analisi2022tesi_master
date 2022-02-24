import math
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import os

from pathintegralanalytics.pathintegralanalytics import markov_chain_fun as mcf
from pathintegralanalytics.pathintegralanalytics import pathIntegralObjects as pio
from pathintegralanalytics.pathintegralanalytics import new_plot_functions as plf
from pathintegralanalytics.pathintegralanalytics import LatticePedSimulation as sim


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


def calc_df_info(df, Num_pid_iniziali=0):
    Num_pid_rimasti = int(mcf.count_pids(df))
    distance_min = round(df['distance'].min())
    distance_mean = round(df['distance'].mean())
    distance_max = round(df['distance'].max())
    Rstep_len_min = round(df['Rstep_len'].min())
    Rstep_len_mean = round(df['Rstep_len'].mean())
    Rstep_len_max = round(df['Rstep_len'].max())

    dict_info = {'Num_pid_rimasti': Num_pid_rimasti}
    if Num_pid_iniziali != 0:
        dict_info.update({'Num_pid_iniziali': Num_pid_iniziali})
    dict_info.update({'distance_min': distance_min
                     , 'distance_mean': distance_mean
                     , 'distance_max': distance_max
                     , 'Rstep_len_min': Rstep_len_min
                     , 'Rstep_len_mean': Rstep_len_mean
                     , 'Rstep_len_max': Rstep_len_max})
    return dict_info


def save_csv(df, source_file_name, processed=True, reduced=True):
    target_file_name = source_file_name[:-4]
    if processed:
        target_file_name = target_file_name + '_proc'
    if reduced:
        target_file_name = target_file_name + '_red'
    target_file_name = target_file_name + '.csv'
    df.to_csv(target_file_name)
    print("\nNew dataframe saved in: \n >>  " + target_file_name)
    return target_file_name


def save_txt_info(dict_info, target_file_name):
    create_txt_file_same_name = target_file_name[:-4] + '.txt'
    creationtime = datetime.datetime.now()
    cosa_scrivere_nel_file = 'CREATION DATE & TIME :  ' + str(creationtime) + \
                             '\n\ninfo txt file of dataframe: ' + \
                             '\n\n >>  ' + str(target_file_name) + \
                             '\n\nDistance minima = ' + str(dict_info['distance_min']) +  \
                             '\nDistance media = ' + str(dict_info['distance_mean']) +  \
                             '\nDistance max = ' + str(dict_info['distance_max']) +  \
                             '\nRstep_len minima = ' + str(dict_info['Rstep_len_min']) +  \
                             '\nRstep_len media = ' + str(dict_info['Rstep_len_mean']) +  \
                             '\nRstep_len max = ' + str(dict_info['Rstep_len_max'])
    if 'Num_pid_iniziali' in dict_info:
        cosa_scrivere_nel_file = cosa_scrivere_nel_file + \
                                 '\nNumero di pid rimasti su iniziali = ' + \
                                 str(dict_info['Num_pid_rimasti']) + \
                                 ' / ' + str(dict_info['Num_pid_iniziali'])
    else:
        cosa_scrivere_nel_file = cosa_scrivere_nel_file + \
                                 '\nNumero di pid rimasti = ' + \
                                 str(dict_info['Num_pid_rimasti'])

    with open(create_txt_file_same_name, 'w') as f:
        f.write(cosa_scrivere_nel_file)

    print("\nNew info.txt saved in: \n >>  " + create_txt_file_same_name)
    return cosa_scrivere_nel_file


#  ---  ---  ---  End of the library  ---  ---  ---





