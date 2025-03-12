from pandas import read_csv, DataFrame
from df_operations import create_dict_kpi
from project_constants import KPIS_CSV_FILE, KPIS_COLUMNS
from topsis import TOPSIS
from analize_kpis import DisplayResults

if __name__ == "__main__":

    SOL_TYPE = [0, 1]  # 0: Non-dominated, 1: dominated
    DAYS = ['2019-03-04', '2019-05-20', '2019-06-10']  # Chose the days to be analyzed
    ALL_KPIs = ['RT_VAL', 'N_RT', 'TOTAL_SLACK', 'N_SWAPS', 'R_DUR']

    # To add scenarios:
    # , {key+1: {'RT_VAL': , 'N_RT':, 'TOTAL_SLACK': , 'N_SWAPS': , 'ABS_DELAY':, 'ACT_HOURS': }}
    # In each scenario, each KPI must point to a positive number between 0 and 1 and the sum of all of them
    # must be 1
    # 1: Prioritizes rotation robustness against flight delays
    # 2: Prioritizes passengers' reservations aiming at minimizing the changes in their original bookings
    # 3: Prioritizes the cots associated with the crew
    weights = {1: {'N_RT': 0,
                   'RT_VAL': 0,
                   'N_SWAPS': 0.5,
                   'TOTAL_SLACK': 0.5,
                   'R_DUR': 0},
               2: {'N_RT': 0.5,
                   'RT_VAL': 0.5,
                   'N_SWAPS': 0,
                   'TOTAL_SLACK': 0,
                   'R_DUR': 0},
               3: {'N_RT': 0,
                   'RT_VAL': 0,
                   'N_SWAPS': 0,
                   'TOTAL_SLACK': 0,
                   'R_DUR': 1}}

    KPIs_To_Labels = ['TOTAL_SLACK(OF_4)', 'N_RT(OF_3)']
    KPIsToAnalyze = ['TOTAL_SLACK', 'N_RT']

    display_results = DisplayResults(kpis_to_analyze=KPIsToAnalyze,
                                     kpis_to_labels=KPIs_To_Labels)

    print('Read kpis data...')
    df_kpis = read_csv(KPIS_CSV_FILE,
                       sep=",",
                       usecols=KPIS_COLUMNS,
                       dtype=create_dict_kpi(KPIS_COLUMNS),
                       header=0)
    df_kpis.rename(columns={'ACT_HOURS': 'R_DUR'}, inplace=True)

    #df_kpis['N_RT'] = df_kpis['N_RT'].multiply(-1)
    #df_kpis['RT_VAL'] = df_kpis['RT_VAL'].multiply(-1)
    #df_kpis['R_DUR'] = df_kpis['R_DUR'].multiply(-1)

    df = (df_kpis.loc[df_kpis['N_SOL'] > 0].loc[df_kpis['DATE'].isin(DAYS)].loc[df_kpis['IS_DOMINATED'].isin(SOL_TYPE)].
          groupby(by=['DATE', 'FLEET']))

    KPIsToPlot = [(k1, k2) for k1 in KPIsToAnalyze for k2 in KPIsToAnalyze
                  if KPIsToAnalyze.index(k1) < KPIsToAnalyze.index(k2)]

    print('Drawing scatter plots...')
    for kpis in KPIsToPlot:
        print(' {} vs {}'.format(kpis[0], kpis[1]))
        for row in df:
            print(' At {} for {}'.format(row[0][0], row[0][1]))
            cols = [kpis[0], kpis[1], 'IS_DOMINATED']
            display_results.KPIs_to_scatter_plot(data=row[1][cols],
                                                 plot_date=row[0][0],
                                                 fleet_type=row[0][1],
                                                 lim_axis=[0, 4500, 130, 260])

    KPIsToAnalyze = ['N_RT', 'RT_VAL', 'N_SWAPS', 'TOTAL_SLACK', 'R_DUR']
    KPIs_To_Labels = ['OF3(N_RT)',
                      'KPI1(RT_VAL)',
                      'KPI2(N_SWAPS)',
                      'OF4(SLACK)',
                      'KPI3(R_DUR)']

    display_results.update_settings(kpis_to_analyze=KPIsToAnalyze,
                                    kpis_to_labels=KPIs_To_Labels)

    # Plot the relationships among all KPIs using a heatmap
    display_results.KPIS_correlation2(data=df[ALL_KPIs])

    #display_results.KPIs_impact_on_solutions(data=df[ALL_KPIs])

    print('Plotting all solutions on a radar chart...')
    data_norm = dict()
    for row in df:
        print(' to day: {} for fleet type: {}'.format(row[0][0], row[0][1]))

        # Plot the impact of each KPI on each given solution using a heatmap
        #display_results.KPIs_impact_on_solutions(data=row[1][ALL_KPIs],
        #                                         plot_date=row[0][0],
        #                                         fleet_type=row[0][1])

        # Plot the relationships among all KPIs using a heatmap
        #display_results.KPIS_correlation(data=row[1][ALL_KPIs],
        #                                 plot_date=row[0][0],
        #                                 fleet_type=row[0][1])

        # Normalizing data
        data_norm.update({(row[0][0], row[0][1]):
                              display_results.normalize_kpis(data=row[1][KPIsToAnalyze].reset_index(drop=True))})

        # Plot normalize data into a radar chart
        display_results.KPIs_to_radar_chart(data=data_norm[(row[0][0], row[0][1])],
                                            plot_date=row[0][0],
                                            fleet_type=row[0][1],
                                            show_legend=False)

    print('Applying TOPSIS...')
    #ben_att = {'N_RT', 'RT_VAL', 'N_SWAPS', 'TOTAL_SLACK', 'R_DUR'}
    ben_att = {'N_SWAPS', 'TOTAL_SLACK'}
    topsis = TOPSIS(KPIsToAnalyze=KPIsToAnalyze,
                    benefit_attributes=ben_att)

    for row in df:
        print(' to day: {} for fleet type: {}'.format(row[0][0], row[0][1]))

        data = row[1].reset_index(drop=True)
        best_sol = DataFrame()
        for w in weights:
            print('  with weights: {}'.format(weights[w]))
            id_sol = topsis.execute(data=data, weights=weights[w], scenario=w)
            best_sol = best_sol.append(data_norm[(row[0][0], row[0][1])].iloc[id_sol][KPIsToAnalyze])

        print('  visualizing TOPSIS results...')
        display_results.KPIs_to_radar_chart(data=best_sol,
                                            plot_date=row[0][0],
                                            fleet_type=row[0][1],
                                            scenario='TOPSIS')
