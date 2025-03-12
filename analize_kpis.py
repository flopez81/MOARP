import matplotlib.pyplot as plt
from numpy import linspace, pi, degrees, array
import plotly.graph_objects as go
from seaborn import heatmap
from pandas import DataFrame
from project_constants import KPIS_CSV_FILE, KPIS_COLUMNS


class DisplayResults:
    def __init__(self, kpis_to_labels: list, kpis_to_analyze: list, axis_labels_size: int = 15,
                 axis_ticks_size: int = 15):
        """
        Initialization method of the class where all its attributes are set up to their initial methods
        Later, they can be updated in update_settings method.
        :param kpis_to_labels: A list of names of the KPIs to be shown in the legend of the plots to be used later
        :param kpis_to_analyze: A list of names of the KPIs to be used as a filters for the columns held in the input
                                DataFrame data of the plot methods
        :param axis_labels_size: An integer used to set the size of the labels attached to x/y-labels
        :param axis_ticks_size: An integer used to set the size of the numbers attached to x/y-labels
        """
        self.ALL_KPIs = ['RT_VAL', 'N_RT', 'TOTAL_SLACK', 'N_SWAPS', 'R_DUR']
        self.KPIsToLabels = kpis_to_labels
        self.KPIsToAnalyze = kpis_to_analyze
        self.AXIS_LABELS_SIZE = axis_labels_size
        self.AXIS_TICKS_SIZE = axis_ticks_size
        self.TRACE_COLORS = ['black', 'grey', 'gold', 'yellowgreen', 'darkolivegreen', 'red', 'darkorange', 'blue',
                             'purple', 'darkcyan', 'peru', 'hotpink', 'darkblue', 'tan', 'violet', 'yellow',
                             'rosybrown', 'darkred', 'darkslategrey', 'olive', 'lime']  # 21

    def update_settings(self, kpis_to_labels: list, kpis_to_analyze: list) -> None:
        """
        This method updates the main attributes of the class
        :param kpis_to_labels: A list of names of the KPIs to be shown in the legend of the plots to be used later
        :param kpis_to_analyze: A list of names of the KPIs to be used as a filters for the columns held in the input
                                DataFrame data of the plot methods
        """
        self.KPIsToLabels = kpis_to_labels
        self.KPIsToAnalyze = kpis_to_analyze

    def KPIs_to_scatter_plot(self, data: DataFrame, plot_date: str, fleet_type: str, lim_axis: list = None) -> None:
        """
        This method plots a couple of given KPIs held in a two-column DataFrame data into a scatter plot
        :param lim_axis: list containing the axis limits (i.e. [x_min, x_max, y_min, y_max])
        :param data: An input two-column DataFrame with the solution
        :param plot_date: A date string with the format year-month-day
        :param fleet_type:  A three-digit string representing the type of aircraft used in all solutions
        """

        plt.figure(figsize=(7, 5.5))

        if lim_axis:
            plt.axis(lim_axis)

        kpis = data.columns
        data1 = data.loc[data['IS_DOMINATED'] == 1]
        plt.scatter(data1[kpis[0]], data1[kpis[1]], color='b', label='pareto')

        data1 = data.loc[data['IS_DOMINATED'] == 0]
        plt.scatter(data1[kpis[0]], data1[kpis[1]], color='r', label='non-pareto')

        plt.xticks(fontsize=self.AXIS_TICKS_SIZE)
        plt.xlabel(self.KPIsToLabels[0], fontsize=self.AXIS_LABELS_SIZE)
        plt.yticks(fontsize=self.AXIS_TICKS_SIZE)
        plt.ylabel(self.KPIsToLabels[1], fontsize=self.AXIS_LABELS_SIZE)
        plt.title(plot_date, fontsize=self.AXIS_LABELS_SIZE)

        date_str = plot_date[:4] + plot_date[5:7] + plot_date[8:10]
        plot_file = kpis[0] + '_vs_' + kpis[1] + '_' + date_str + '_' + fleet_type + ".png"

        plt.savefig(plot_file, dpi=150)
        plt.close()

    def normalize_kpis(self, data: DataFrame) -> DataFrame:
        """
        This method normalizes the data within [0, 1] interval given all the solutions that can be plotted later
        using some of the available plot methods
        :param data: An input DataFrame holding all the solutions for a given date/fleet type and for all KPIs
        :return: An output DataFrame with the normalized data
        """

        data_norm = data.copy()

        for kpi in data.columns:
            print('Normalizing {}'.format(kpi))

            # 1.Calculate the range of the data set
            min_val = data[kpi].min()
            max_val = data[kpi].max()

            range_val = max_val - min_val

            # 2. Subtract the minimum x value from each value of the data set
            data_norm[kpi] = data[kpi].sub(min_val).fillna(0)

            # 3.Divide by the range
            data_norm[kpi] = data_norm[kpi].div(range_val).fillna(0)

        return data_norm

    def KPIs_to_radar_chart_interactive(self, data: DataFrame, plot_date: str, fleet_type: str) -> None:
        """
        This method plots the solutions held on Data into an interactive radar chart where each dimension
        represents a KPI attached to a column in Data
        :param data: An input DataFrame with rows representing each solution and columns standing for each KPI
        :param plot_date: A date string with the format year-month-day
        :param fleet_type: A three-digit string representing the type of aircraft used in all solutions
        """

        kpis = data.columns

        fig = go.Figure()
        for i, row in data.iterrows():
            r = row[kpis].tolist()
            fig.add_trace(go.Scatterpolar(
                r=[*r, r[0]],
                theta=[*kpis, kpis[0]],
                fill='none',  # toself
                name='Sol' + str(i + 1)
            ))

        # date_str = plot_date[:4] + plot_date[5:7] + plot_date[8:10]

        fig.update_layout(
            title=dict(text=plot_date + ' for fleet ' + fleet_type),
            # polar=dict(
            #    radialaxis=dict(
            #        visible=True,
            #        range=[0, 1]
            #    )),
            showlegend=True
        )

        fig.show()
        # fig.write_image(date_str + '_' + fleet_type + ".png")

    def KPIs_to_radar_chart(self, data: DataFrame, plot_date: str, fleet_type: str, scenario: str = 'all',
                            show_legend: bool = True) -> None:
        """
        This method plots the solutions held on Data into a radar chart where each dimension represents a KPI
        attached to a column in Data
        :param data: An input DataFrame with rows representing each solution and columns standing for each KPI
        :param plot_date: A date string with the format year-month-day
        :param fleet_type: A three-digit string representing the type of aircraft used in all solutions
        :param scenario: A name describing the type of scenario under which the solutions have been evaluated
        :param show_legend: A boolean indicating if the legend will be shown (by default the legend is shown)
        """

        data_sorted = data.reindex(sorted(data.columns), axis=1).copy()
        kpis = data_sorted.columns
        label_loc = linspace(start=0, stop=2 * pi, num=len(kpis) + 1)
        plt.figure(figsize=(8.5, 8))
        plt.subplot(polar=True)
        plt.xticks(fontsize=self.AXIS_TICKS_SIZE)
        plt.yticks(fontsize=self.AXIS_TICKS_SIZE)

        id_row = 1
        for i, row in data_sorted.iterrows():
            r = row[kpis].tolist()
            plt.plot(label_loc, [*r, r[0]], color=self.TRACE_COLORS[i], label='Criterion ' + str(id_row))
            id_row += 1

        plt.title(plot_date, fontsize=self.AXIS_TICKS_SIZE)
        lines, labels = plt.thetagrids(degrees(label_loc), labels=[*self.KPIsToLabels, self.KPIsToLabels[0]],
                                       fontsize=self.AXIS_TICKS_SIZE)

        if show_legend:
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc='lower center', ncol=3, fontsize=self.AXIS_TICKS_SIZE,
                       frameon=False)

        date_str = plot_date[:4] + plot_date[5:7] + plot_date[8:10]
        plt.savefig('radarchart_' + date_str + '_' + fleet_type + '_' + scenario + ".png", dpi=150)
        plt.close()

    def KPIS_correlation(self, data: DataFrame, plot_date: str = None, fleet_type: str = None,
                         corr_method: str = 'pearson') -> None:
        """
        This method shows the correlation matrix of the given KPIs (related to each column in the input DataFrame)
        considering all the solutions held in each row of the input DataFrame using a heatmap
        :param corr_method:
        :param data: An input DataFrame with rows representing each solution and columns standing for each KPI
        :param plot_date: A date string with the format year-month-day
        :param fleet_type: A three-digit string representing the type of aircraft used in all solutions
        """

        # Computing the correlation matrix
        corr_matrix = data.corr(method=corr_method).fillna(0)

        plt.figure(figsize=(18, 12))
        #plt.figure(figsize=2 * array(plt.rcParams['figure.figsize']))

        # Options used:
        # cbar=True: The vertical bar showing the color density is diplayed
        # Annot=True: shows the values in the correlation matrix cells
        # Further info at https://seaborn.pydata.org/generated/seaborn.heatmap.html
        heatmap(data=corr_matrix, cbar=True, annot=True)

        plt.xticks(rotation=0)
        plt.yticks(rotation=0)

        # plt.imshow(corr_matrix, cmap="RdYlBu")

        # Displaying a color bar to understand which color represents which range of data
        # plt.colorbar()

        # Assigning labels of x-axis according to dataframe
        # plt.xticks(range(len(corr_matrix)), corr_matrix.columns)

        # Assigning labels of y-axis according to dataframe
        # plt.yticks(range(len(corr_matrix)), corr_matrix.index)

        # plt.xlabel('KPIs')
        # plt.ylabel('KPIs')
        if plot_date is not None or fleet_type is not None:
            plt.title(plot_date + ' for fleet ' + fleet_type)
            date_str = plot_date[:4] + plot_date[5:7] + plot_date[8:10]
            plt.savefig('corr_heatmap_' + date_str + '_' + fleet_type + ".png", dpi=150)
        else:
            plt.savefig('corr_heatmap.png', dpi=150)

        plt.close()

    def KPIS_correlation2(self, data: DataFrame, corr_method: str = 'pearson', fontsize: int = 6,
                          show_annot: bool = True) -> None:
        """
        This method shows the correlation matrix of the given KPIs (related to each column in the input DataFrame)
        considering all the solutions held in each row of the second dimension of the grouped DataFrame using a heatmap.
        Each solution is displayed horizontally in a subplot. All subplot shares the same x-y-axis
        :param corr_method:
        :param fontsize:
        :param show_annot:
        :param data: A grouped DataFrame with rows in the second dimension representing each solution and columns
                     standing for each KPI. In the first dimension, there is a tuple with the date and type of aircraft
                     being used in the solutions.
        :corr_method: The type of method (statistic measure) to use to analyze the relationship among KPIs.
                      By default, it is used the pearson method
        """

        plt.figure(figsize=(30, 25))

        n_scenarios = len(data)

        f, axis = plt.subplots(1, n_scenarios + 1, gridspec_kw={'width_ratios': [1] * n_scenarios + [0.08]})

        axis[0].get_shared_y_axes().join(axis[i] for i in range(1, n_scenarios))

        g = list()
        i = 0
        for row in data:

            # Computing the correlation matrix
            corr_matrix = row[1][self.ALL_KPIs].corr(method=corr_method).fillna(0)

            if i < n_scenarios - 1:
                g.append(heatmap(data=corr_matrix, cmap="YlGnBu", cbar=False, ax=axis[i], annot=show_annot,
                                 annot_kws={"fontsize": fontsize}))
            else:
                g.append(heatmap(data=corr_matrix, cmap="YlGnBu", ax=axis[i], cbar_ax=axis[i + 1], annot=show_annot,
                                 annot_kws={"fontsize": fontsize}))
                axis[i + 1].tick_params(axis='both', which='major', labelsize=fontsize)

            g[i].set_ylabel('')
            g[i].set_xlabel('')

            axis[i].title.set_text(row[0][0])
            axis[i].title.set_size(fontsize)

            axis[i].tick_params(axis='both', which='major', labelsize=fontsize)

            if i > 0:
                g[i].set_yticks([])

            i += 1

        # Rotate the ticklabels correctly
        for ax in g:
            tl = ax.get_xticklabels()
            ax.set_xticklabels(tl, rotation=45)
            tly = ax.get_yticklabels()
            ax.set_yticklabels(tly, rotation=0)

        plt.savefig('corr_heatmap.png', dpi=150)
        plt.close()

    def KPIs_impact_on_solutions(self, data: DataFrame, plot_date: str = None, fleet_type: str = None) -> None:
        """
        This method shows the impact of each KPI (related to a column in the input DataFrame) on each solution held
        in a row in the input DataFrame using a heatmap
        :param data: An input DataFrame with rows representing each solution and columns standing for each KPI
        :param plot_date: A date string with the format year-month-day
        :param fleet_type: A three-digit string representing the type of aircraft used in all solutions
        """

        plt.figure(figsize=2.25 * array(plt.rcParams['figure.figsize']))

        # Options used:
        # cbar=True: The vertical bar showing the color density is displayed
        # Annot=True: shows the values in the correlation matrix cells
        # Further info at https://seaborn.pydata.org/generated/seaborn.heatmap.html
        heatmap(data=data, cbar=True, annot=True)

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        if plot_date is not None or fleet_type is not None:
            plt.title(plot_date + ' for fleet ' + fleet_type)
            date_str = plot_date[:4] + plot_date[5:7] + plot_date[8:10]
            plt.savefig('kpis_heatmap_' + date_str + '_' + fleet_type + ".png", dpi=150)
        else:
            plt.savefig('kpis_heatmap.png', dpi=150)

        plt.close()
