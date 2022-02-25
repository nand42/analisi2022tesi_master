# from pathintegralanalytics.pathintegralanalytics import new_plot_functions as plf

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import pandas as pd
from pathintegralanalytics.pathintegralanalytics import markov_chain_fun as mcf
from pathintegralanalytics.pathintegralanalytics import pathIntegralObjects as pio
import trainf10_operativefile as op


class PlotPDF(object):
    def __init__(self, df, plot_par, **kw):
        self._verbose = False
        self.df = df
        self.par = plot_par
        self.space_type = 'grid'
        self.max_x = 0
        self.max_y = 0
        self.x_ax = 'x_SG'
        self.y_ax = 'y_SG'
        self.x_bins = None
        self.y_bins = None
        self.dpi = 100
        self.bins_in_plot_scale = 10
        self.bins_in_plot_position_heatmap = 5
        self.max_velocity = 2
        self.grb = 'pid'
        self.experiment = self.par['experiment']
        self.datatype = self.par['datatype']
        self.format = self.par['format']
        self.target = 'figure.pdf'
        self.name_plots()
        self.title = "Cool figure"
        self.x_label = 'x - direction'
        self.y_label = 'y - direction'
        self.clear_figure = True
        self.SG_coordinates = True
        self.standard_cmap = 'Greens'

    def define_physic_space(self):
        """
        if self.par['space_type'] == 'grid':
            self.space_type = 'grid'
        elif self.par['space_type'] == 'continuous':
            self.space_type = 'continuous'
        else:
            self.space_type = 'grid'
        """
        if self.space_type == 'grid':
            self.max_x = self.par['Dx']
            self.max_y = self.par['Dy']
            return 'grid'
        elif self.space_type == 'continuous':
            self.max_x = self.par['Lx']
            self.max_y = self.par['Ly']
            return 'continuous'
        else:
            return 'None'

    def define_velocity_space(self):
        if self.space_type == 'grid':
            self.max_x = self.max_velocity
            self.max_y = self.max_velocity
            return 'grid'
        elif self.space_type == 'continuous':
            self.max_x = self.max_velocity
            self.max_y = self.max_velocity
            return 'continuous'
        else:
            return 'None'

    def change_title(self, new_title='None'):
        if new_title != 'None':
            self.title = new_title
        else:
            print('Actual pdf title ' + str(self.title))
            self.title = input('Set new title: ')

    def change_x_label(self, new_x_label='None'):
        if new_x_label != 'None':
            self.x_label = new_x_label
        else:
            print('Actual x label ' + str(self.x_label))
            self.x_label = input('Set new x label: ')

    def change_y_label(self, new_y_label='None'):
        if new_y_label != 'None':
            self.y_label = new_y_label
        else:
            print('Actual y label ' + str(self.y_label))
            self.y_label = input('Set new y label: ')

    def change_xy(self, xax='None', yax='None'):
        self.SG_coordinates = False

        if xax == 'None':
            self.x_ax = input("Specify x axes: ")
            self.max_x = round(abs(max(self.df.eval(self.x_ax))))
            if self.max_x == 0:
                self.max_x += 1
            if self._verbose: print('max_x = ' + str(self.max_x))
        else:
            self.x_ax = xax
            self.max_x = round(abs(max(self.df.eval(self.x_ax))))
            if self.max_x == 0:
                self.max_x += 1
            if self._verbose: print('max_x = ' + str(self.max_x))

        if yax == 'None':
            self.y_ax = input("Specify x axes: ")
            self.max_y = round(abs(max(self.df.eval(self.y_ax))))
            if self.max_y == 0:
                self.max_y += 1
            if self._verbose: print('max_y = ' + str(self.max_y))
        else:
            self.y_ax = yax
            self.max_y = round(abs(max(self.df.eval(self.y_ax))))
            if self.max_y == 0:
                self.max_y += 1
            if self._verbose: print('max_y = ' + str(self.max_y))

    def _change_file_name_end(self, add_text=''):
        add_text = '_' + add_text
        self.target = self.target[:-4] + str(add_text) + str(self.format)
        if self._verbose: print('[_change_file_name_end] -- finished')

    def change_groupby(self):
        new_selgrb = input("Set new groupby as = ")
        self.grb = str(new_selgrb)

    def name_plots(self, figtype='none'):
        self.target = 'figure_' + self.experiment + '_' + self.datatype + '_' + figtype + self.format

    def _set_plot_positions(self, heatmap=False):
        self.space_type = self.define_physic_space()
        dx = self.bins_in_plot_scale
        dy = self.bins_in_plot_scale
        self.x_bins = np.arange(0, self.max_x + dx,  dx)
        self.y_bins = np.arange(0, self.max_y + dy,  dy)

        if self._verbose:
            print('[_set_plot_positions] -- MAX x y')
            print(self.max_x, self.max_y)

        r = self.max_x / self.max_y
        dy = 10
        dx = r * dy
        if self._verbose: print('[_set_plot_positions] -- start plt.figure')
        plt.figure(figsize=(dx, dy), dpi=self.dpi)

        plt.xlim([0, self.max_x])
        plt.ylim([0, self.max_y])

        if heatmap:
            dx = self.bins_in_plot_position_heatmap
            dy = self.bins_in_plot_position_heatmap
            self.x_bins = np.arange(0, self.max_x + dx, dx)
            self.y_bins = np.arange(0, self.max_y + dy, dy)

        if self._verbose: print('[_set_plot_positions] -- start plt.xticks and plt.yticks')
        plt.xticks(self.x_bins)
        plt.yticks(self.y_bins)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        if self._verbose: print('[_set_plot_positions] -- finished')

    def _set_plot_velocities(self):
        self.define_velocity_space()
        dx = 0.1
        dy = 0.1
        self.x_bins = np.arange(-self.max_x, self.max_x + dx,  dx)
        self.y_bins = np.arange(-self.max_y, self.max_y + dy,  dy)

        dy = 10 * self.max_velocity
        dx = 10 * self.max_velocity

        plt.figure(figsize=(dx, dy), dpi=self.dpi)

        plt.xlim([-self.max_velocity, self.max_velocity])
        plt.ylim([-self.max_velocity, self.max_velocity])

        plt.xticks(self.x_bins)
        plt.yticks(self.y_bins)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)


class PositionLines(PlotPDF):
    def __init__(self, df, plot_par, **kw):
        super().__init__(df, plot_par, **kw)
        self.plot_type = 'lines'
        self.name_plots(figtype=self.plot_type)
        self.title = 'Trajectories lines'

    def figure_save(self, add_info_to_end_path='None'):
        if self.clear_figure:
            plt.clf()
        self._set_plot_positions()
        if add_info_to_end_path != 'None':
            self._change_file_name_end(add_text=add_info_to_end_path)
        if self._verbose: print('[figure_save] -- start plt.plot')
        self.df.groupby(self.grb).apply(lambda x: plt.plot(x.eval(self.x_ax), x.eval(self.y_ax)))
        if self._verbose: print('[figure_save] -- start plt.savefig')
        plt.savefig(self.target)
        if self._verbose: print('[figure_save] -- finished')


class PositionHeatmap(PlotPDF):
    def __init__(self, df, plot_par, **kw):
        super().__init__(df, plot_par, **kw)
        self.space_type = self.define_physic_space()
        self.plot_type = 'heatmap'
        self.name_plots(figtype=self.plot_type)
        self.title = 'Trajectories heatmap'

    def figure_save(self, add_info_to_end_path='None', logscale=True):
        self._set_plot_positions(heatmap=True)
        if logscale:
            plt.hist2d(self.df.eval(self.x_ax)
                       , self.df.eval(self.y_ax)
                       , bins=(self.x_bins, self.y_bins)
                       , norm=LogNorm()
                       , cmap=self.standard_cmap)
        else:
            plt.hist2d(self.df.eval(self.x_ax)
                       , self.df.eval(self.y_ax)
                       , bins=(self.max_x, self.max_y)
                       , cmap=self.standard_cmap)
        plt.colorbar()
        if add_info_to_end_path != 'None':
            self._change_file_name_end(add_text=add_info_to_end_path)
        plt.savefig(self.target)


class VelocityHeatmap(PlotPDF):
    def __init__(self, df, plot_par, **kw):
        super().__init__(df, plot_par, **kw)
        self.plot_type = 'velocity_heatmap'
        self.name_plots(figtype=self.plot_type)
        self.title = 'Velocity heatmap'
        self.change_x_label(new_x_label='x velocity')
        self.change_y_label(new_y_label='y velocity')
        self.change_xy(xax='vx_SG', yax='vy_SG')

    def figure_save(self, add_info_to_end_path='None', logscale=True):
        if self.clear_figure:
            plt.clf()
        self._set_plot_velocities()
        if logscale:
            plt.hist2d(self.df.eval(self.x_ax)
                       , self.df.eval(self.y_ax)
                       , bins=(self.x_bins, self.y_bins)
                       , norm=LogNorm()
                       , cmap=self.standard_cmap)
        else:
            plt.hist2d(self.df.eval(self.x_ax)
                       , self.df.eval(self.y_ax)
                       , bins=(self.x_bins, self.y_bins)
                       , cmap=self.standard_cmap)
        plt.colorbar()
        if add_info_to_end_path != 'None':
            self._change_file_name_end(add_text=add_info_to_end_path)
        plt.savefig(self.target)
        return self.target


def select_pids(df, pid_list, _verbose=False):
    if _verbose: print(df[:3])

    df_SomePids = df[0:0]

    if _verbose: print(df_SomePids[:3])

    for pid_selection in pid_list:
        if _verbose: print('\nselected pid ' + str(pid_selection))
        to_append = df.query('tracked_object == @pid_selection')
        if _verbose: print(to_append[:3])
        df_SomePids = pd.concat([df_SomePids, to_append])

    return df_SomePids


def make_basic_plots_OnePid(source_file_path, par, pid, _verbose=False):
    experiment = par['experiment']
    what = 'RealData'
    format = par['format']
    pid_selection = pid
    target_file_path = source_file_path[:-4] + '_OnePid.csv'

    df = pd.read_csv(source_file_path)
    df_OnePid = df.query('tracked_object == @pid_selection')
    length = len(df_OnePid)
    target_file_path_ID = target_file_path[:-4] + "_" + str(pid_selection) + "_len_" + str(length) + ".csv"

    df_OnePid.to_csv(target_file_path_ID)
    mcf.count_pids(df_OnePid, print_=True, grb='tracked_object')

    print(df_OnePid[:6])
    print("Selected One Pid with \nID = " + str(pid_selection) + " \nlenght = " + str(length))
    print("New dataframe with One Pid saved in " + target_file_path_ID)

    # --- start second part ---
    if _verbose: print('-a-')
    file_list = [target_file_path_ID]
    if _verbose: print('-b-')
    target_file_path = target_file_path_ID[:-4] + "_processed" + ".csv"
    if _verbose: print('-c-')

    pedDataIface = pio.factory_PedestrianTrajectoryDataInterface(file_list, par)
    if _verbose: print('-d-')
    pedDataIface.calculate_standard_df_override()
    if _verbose: print('-e-')
    pedDataIface.calculate_velocity_xy()
    if _verbose: print('-f-')
    print(pedDataIface.df[:6])

    print('\nmin - max of x_SG')
    print(min(pedDataIface.df.x_SG), max(pedDataIface.df.x_SG))
    print('\nmin - max of y_SG')
    print(min(pedDataIface.df.y_SG), max(pedDataIface.df.y_SG))

    print('\nmin - max of vx')
    print(min(pedDataIface.df.vx_SG), max(pedDataIface.df.vx_SG))
    print('\nmin - max of vy')
    print(min(pedDataIface.df.vy_SG), max(pedDataIface.df.vy_SG))

    print('\nmin - max of Rstep')
    print(min(pedDataIface.df.Rstep), max(pedDataIface.df.Rstep))

    df = pedDataIface.df
    df.to_csv(target_file_path)

    print(df[:6])
    print("Selected One Pid with \nID = " + str(pid_selection) + " \nlenght = " + str(length))
    print("New dataframe with One Pid saved in " + target_file_path_ID)

    # --- start make figures ---

    print('\nPlotting figures ... it could takes a while: \n')

    lines = PositionLines(df, par)
    lines.figure_save(add_info_to_end_path=str(pid_selection))
    print("ok immagine 1")

    heatmap = PositionHeatmap(df, par)
    heatmap.figure_save(add_info_to_end_path=str(pid_selection))
    print("ok immagine 2")

    velheatmap = VelocityHeatmap(df, par)
    velheatmap.figure_save(add_info_to_end_path=str(pid_selection))
    print("ok immagine 3")


def make_basic_plots(source_file_path, par, _verbose=False):
    experiment = par['experiment']
    what = 'RealData'
    format = par['format']

    target_file_path = source_file_path[:-4] + '_OnePid.csv'

    df = pd.read_csv(source_file_path)

    pid_number = mcf.count_pids(df, print_=True, grb='tracked_object')
    target_file_path_ID = target_file_path[:-4] + "_PidNum_" + str(pid_number) + ".csv"
    df.to_csv(target_file_path_ID)

    print(df[:6])
    print("Selected dataframe has \na number of pid = " + str(pid_number))
    print("New dataframe with One Pid saved in " + target_file_path_ID)

    # --- start second part ---
    if _verbose: print('-a-')
    file_list = [target_file_path_ID]
    if _verbose: print('-b-')
    target_file_path = target_file_path_ID[:-4] + "_processed" + ".csv"
    if _verbose: print('-c-')

    pedDataIface = pio.factory_PedestrianTrajectoryDataInterface(file_list, par)
    if _verbose: print('-d-')
    pedDataIface.calculate_standard_df_override()
    if _verbose: print('-e-')
    pedDataIface.calculate_velocity_xy()
    if _verbose: print('-f-')
    print(pedDataIface.df[:6])

    print('\nmin - max of x_SG')
    print(min(pedDataIface.df.x_SG), max(pedDataIface.df.x_SG))
    print('\nmin - max of y_SG')
    print(min(pedDataIface.df.y_SG), max(pedDataIface.df.y_SG))

    print('\nmin - max of vx')
    print(min(pedDataIface.df.vx_SG), max(pedDataIface.df.vx_SG))
    print('\nmin - max of vy')
    print(min(pedDataIface.df.vy_SG), max(pedDataIface.df.vy_SG))

    print('\nmin - max of Rstep')
    print(min(pedDataIface.df.Rstep), max(pedDataIface.df.Rstep))

    df = pedDataIface.df
    df.to_csv(target_file_path)

    print(df[:6])
    print("Selected dataframe has \na number of pid = " + str(pid_number))
    print("New dataframe with One Pid saved in " + target_file_path_ID)

    # --- start make figures ---

    print('\nPlotting figures ... it could takes a while: \n')

    lines = PositionLines(df, par)
    lines.figure_save(add_info_to_end_path=str(pid_number))
    print("ok immagine 1")

    heatmap = PositionHeatmap(df, par)
    heatmap.figure_save(add_info_to_end_path=str(pid_number))
    print("ok immagine 2")

    velheatmap = VelocityHeatmap(df, par)
    velheatmap.figure_save(add_info_to_end_path=str(pid_number))
    print("ok immagine 3")


def just_plot_three(source_file_path, par, add_info='', _verbose=False):

    df = pd.read_csv(source_file_path)

    pid_number = mcf.count_pids(df, print_=True, grb='pid')
    add_info = add_info + '_NumP_' + str(pid_number) + '_'

    if _verbose: print('DF has ' + str(pid_number) + ' pids')

    # --- start make figures ---

    if _verbose: print('\nPlotting figures ... it could takes a while: \n')

    lines = PositionLines(df, par)
    lines.figure_save(add_info_to_end_path=add_info)
    if _verbose: print("ok immagine 1")

    heatmap = PositionHeatmap(df, par)
    heatmap.figure_save(add_info_to_end_path=add_info)
    if _verbose: print("ok immagine 2")

    velheatmap = VelocityHeatmap(df, par)
    folder_target = velheatmap.figure_save(add_info_to_end_path=add_info)
    if _verbose: print("ok immagine 3")
    folder_target = folder_target[:-4]
    return folder_target


def just_plot_three_figeverypid(source_file_path, par, cut_list=0, add_info='', _verbose=False):
    if _verbose: print("[just_plot_three_figeverypid] - start")
    df = pd.read_csv(source_file_path)

    pid_number = mcf.count_pids(df, print_=True, grb='pid')
    add_info = add_info + '_NP_' + str(pid_number) + '_'

    if _verbose: print('DF has ' + str(pid_number) + ' pids')

    pid_list, num_pid = op.make_pid_list_and_count(df)
    if cut_list != 0:
        pid_list = pid_list[:cut_list]
        if _verbose: print('\nList of pids cutted at number of pids = ' + str(num_pid))

    # --- start make figures ---

    for pid in pid_list:
        if _verbose: print('[for loop] - Pid ' + str(pid))

        df_pid = df.query('pid == @pid')
        add_info_pid = add_info + str(pid)

        lines = PositionLines(df_pid, par)
        lines.figure_save(add_info_to_end_path=add_info_pid)
        if _verbose: print("ok immagine 1")

        heatmap = PositionHeatmap(df_pid, par)
        heatmap.figure_save(add_info_to_end_path=add_info_pid)
        if _verbose: print("ok immagine 2")

        velheatmap = VelocityHeatmap(df_pid, par)
        velheatmap.figure_save(add_info_to_end_path=add_info_pid)
        if _verbose: print("ok immagine 3")

    if _verbose: print("[just_plot_three_figeverypid] - finish")


def make_D2Q9_matrix_heatmap(nA, filename='', standard_cmap='Greens', _verbose=False):
    """
    # cmap='Greens'
    # cmap='hot'
    # cmap='binary'
    # cmap='YlGnBu'
    """
    data_type = 'D2Q9'
    map_k = np.array((6, 2, 5, 3, 0, 1, 7, 4, 8))
    if _verbose:
        print("ok")
    plt.clf()
    i = -1
    xmax = np.shape(nA[0, :, 0])[0]
    ymax = np.shape(nA[0, 0, :])[0]
    if _verbose:
        print(xmax, ymax)
    dx = xmax / ymax * 10
    dy = 10
    fig, axes = plt.subplots(nrows=3, ncols=3
                             , sharex=True, sharey=True
                             , figsize=(dx, dy), dpi=1000)

    for ax in axes.flat:
        i += 1
        if _verbose:
            print("ok")

        k = map_k[i]

        im = ax.imshow(np.transpose(nA[k, :, :])
                       , cmap=standard_cmap
                       , vmin=-0.01
                       , interpolation='nearest'
                       , extent=[0, xmax, ymax, 0])

        plt.gca().invert_yaxis()
        ax.set_title('k = ' + str(k))
        ax.axes.xaxis.set_visible(True)
        ax.axes.yaxis.set_visible(True)
        ax.set_xlabel('x')  # range of values in edges
        ax.set_ylabel('y')  # range of values in edges

    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig(filename + "_" + str(data_type) + "nA_" + str('all_3x3') + '.pdf')
    print('\n[all k of D2Q9] - immagine salvata.\n')

    return fig, axes


def make_D2Q9_1by1_heatmap(nA, filename='', standard_cmap='Greens', _verbose=False):
    print('TODO')
    return 'TODO'
