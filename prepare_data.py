from df_operations import create_dict
from numpy import int8, uint8, uint16
from datetime import datetime, timedelta
from pandas import read_csv, DataFrame, to_datetime, Grouper, merge
from project_constants import FLIGHTS_CSV_FILE, FLIGHTS_COLUMNS, PREPARE_FLIGHTS_COLS, FILLED_FLIGHTS_COLS, MIN_TAT, \
    DATA_CSV_FILE, RET_VALUES, PLAN_HOR, PLAN_FLEET, CONNECTIONS_CSV_FILE, FLIGHT_COLS_4_CONN, ETL_FLIGHT_COL, \
    DATE_COLS, GROUPBY_COLS_TO_OPT, REMOVE_COLS, METHOD_COLS, DICT_FLIGHTS_COLUMNS, FLIGHTS_COLUMNS3, AIRLINE, \
    SCENARIOS_CSV_FILE, SCENARIO_OP, FLEET_2_BASES, group_by_week, DATETIME_COLUMNS, ETL_DICT_KEYS, ETL_CSV_FILE


def read_flights_from_csv(*, filename: str) -> DataFrame:
    print('Read flights data')

    flights_types_dict = create_dict(FLIGHTS_COLUMNS)
    input_flights = read_csv(filename,
                             sep=",",
                             usecols=FLIGHTS_COLUMNS,
                             dtype=flights_types_dict,
                             parse_dates=DATETIME_COLUMNS,
                             header=0)

    return input_flights


def prepare_retimings(*, selected_flights: DataFrame) -> DataFrame:

    for c in DATE_COLS:
        selected_flights[c] = to_datetime(selected_flights[c], format='%d/%m/%Y %H:%M').copy()

    final_flights = {c: list() for c in DICT_FLIGHTS_COLUMNS}

    # Selecting the best method (i.e., the one with the least number of unscheduled flights) for each scenario
    # (date, fleet)
    grouped_flights = selected_flights.groupby(GROUPBY_COLS_TO_OPT)
    for row in grouped_flights:
        n_unsch_f = [uint16(row[1][r].sum()) for r in REMOVE_COLS]
        min_n_unsch_f = min(n_unsch_f)
        best_method = n_unsch_f.index(min_n_unsch_f)

        for c in FLIGHTS_COLUMNS3:
            final_flights[c].extend(row[1][c])

        final_flights['rotation'].extend(row[1][METHOD_COLS[best_method][0]])
        final_flights['remove'].extend(row[1][METHOD_COLS[best_method][1]])
        final_flights['conn'].extend(row[1][METHOD_COLS[best_method][2]])

    df_final_flights = DataFrame(final_flights)
    df_final_flights = df_final_flights.astype({'rotation': 'uint8'})

    return df_final_flights


def compute_retimings(*, flights: DataFrame, retimings: list) -> DataFrame:

    retimed_flights = {c: list() for c in PREPARE_FLIGHTS_COLS}

    for i, f in flights.iterrows():
        for r in retimings:
            for c in FILLED_FLIGHTS_COLS:
                retimed_flights[c].append(f[c])

            retimed_flights['local_dep_datetime'][-1] = retimed_flights['local_dep_datetime'][-1] + timedelta(minutes=r)
            retimed_flights['local_arr_datetime'][-1] = retimed_flights['local_arr_datetime'][-1] + timedelta(minutes=r)
            retimed_flights['Delta'].append(uint8(r))

    df_retimed_flights = DataFrame(retimed_flights)

    return df_retimed_flights


def compute_connections(*, flight_retimings: DataFrame) -> DataFrame:

    # Step 1. Create a dummy column to conduct a merge with itself
    flight_retimings['dummy'] = uint8(1)

    # Step 2. Conduct the merge
    df_day = merge(flight_retimings, flight_retimings, on='dummy')

    # Step 3. Filter the dataframe rows satisfying connection conditions
    df_day = df_day.loc[(df_day['Flight No_x'] != df_day['Flight No_y']) &
                        (df_day['Dep Airport Code_y'] == df_day['Arr Airport Code_x']) &
                        (df_day['local_dep_datetime_y'] >= df_day['local_arr_datetime_x'] +
                         timedelta(minutes=MIN_TAT))]

    # Step 4. Compute the extra slack for each feasible connection
    df_day['SLACK'] = uint16((df_day['local_dep_datetime_y'] -
                             (df_day['local_arr_datetime_x'] + timedelta(minutes=MIN_TAT))).dt.total_seconds()/60)

    # Step 5. Append the current feasible connections to the results dataframe
    df_connections = df_day[['Time series_x', 'Specific Aircraft Code_x',
                             'Flight No_x', 'Flight No_y', 'SLACK', 'Delta_x', 'Delta_y']]

    # Step 3. Give the correct names to the columns of the results dataframe
    df_connections.rename(columns={'Time series_x': 'Time series',
                                   'Specific Aircraft Code_x': 'Specific Aircraft Code',
                                   'Flight No_x': 'FLIGHT', 'Flight No_y': 'FLIGHT_CONNECTED',
                                   'Delta_x': 'DELTA',
                                   'Delta_y': 'DELTA_CONNECTED'}, inplace=True)

    return df_connections


def test_call_1(*, airline: str, plan_fleet: list, li: list):
    print('Reading flights data...')
    print("test_call_1 is been called")
    flights = read_flights_from_csv(filename=DATA_CSV_FILE,
                                    first_date=li[0],
                                    last_date=li[1],
                                    plan_fleet=plan_fleet,
                                    airlinename=airline)

    print('Adding re-timings to flights...')
    rt_flights = compute_retimings(flights=flights,
                                   retimings=RET_VALUES)

    print('Computing flights connections...')
    compute_connections(flights=rt_flights)


def prepare_scenarios(*, filename: str, planning_dates: list, fleet_types: list, airlines: list,
                        retimings: list):

    mode = 'w'
    etl_dict = {key: list() for key in ETL_DICT_KEYS}
    for i, plan in enumerate(planning_dates):
        input_flights = read_flights_from_csv(filename=filename)

        print('Filtering flights data from {} to {}'.format(plan[0], plan[1]))
        input_flights = input_flights.loc[input_flights['Time series'].between(plan[0], plan[1]) &
                                          input_flights['Specific Aircraft Code'].isin(fleet_types) &
                                          input_flights['Carrier Code'].isin(airlines)]

        input_flights = input_flights.groupby(by=['Time series', 'Specific Aircraft Code'])
        for row in input_flights:
            print('Day {} and fleet type {}'.format(row[0][0], row[0][1]))
            etl_dict['WEEK'].append(i)
            etl_dict['DATE'].append(row[0][0])
            etl_dict['FLEET'].append(row[0][1])

            print(' Transforming data...')
            flights = prepare_retimings(selected_flights=row[1])

            if flights is None:
                print(' Dataframe is empty!')
                continue

            print(' Adding re-timings to flights...')
            prev = datetime.now()
            rt_flights = compute_retimings(flights=flights, retimings=retimings)

            fr_time = round((datetime.now() - prev).total_seconds()/60, 2)
            etl_dict['N_F_RETS'].append(rt_flights.shape[0])
            etl_dict['T_F_RETS'].append(fr_time)

            print(' Computing flights connections...')
            prev = datetime.now()
            connections = compute_connections(flight_retimings=rt_flights)
            conn_time = round((datetime.now() - prev).total_seconds()/60, 2)
            etl_dict['N_CONN'].append(connections.shape[0])
            etl_dict['T_CONN'].append(conn_time)

            print(' Checking re-timings connections...')
            prev = datetime.now()
            connections["out_conn"] = connections[['Time series', 'Specific Aircraft Code',
                                                   'FLIGHT_CONNECTED', 'DELTA_CONNECTED']].apply(tuple, axis='columns')
            out_conn = set(connections["out_conn"].tolist())
            connections["in_conn"] = connections[['Time series', 'Specific Aircraft Code',
                                                  'FLIGHT', 'DELTA']].apply(tuple, axis='columns')
            in_conn = set(connections["in_conn"].tolist())
            valid_flights = in_conn.union(out_conn)

            if len(in_conn) != len(valid_flights):
                print(' There are {} re-timings do not reachable from others'.format(len(valid_flights) - len(in_conn)))

            if len(out_conn) != len(valid_flights):
                print(' There are {} re-timings unable to reach others'.format(len(valid_flights) - len(out_conn)))

            rt_flights['re_timings'] = rt_flights[['Time series', 'Specific Aircraft Code',
                                                   'Flight No', 'Delta']].apply(tuple, axis='columns')
            final_flights = set(rt_flights["re_timings"].tolist())

            if len(valid_flights) < len(final_flights):
                print(' There are {} re-timings disconnected'.format(len(final_flights) - len(valid_flights)))

            rt_flights.query("re_timings in @valid_flights", inplace=True)
            rt_flights[['Time series', 'Specific Aircraft Code', 'Flight No',
                        'local_dep_datetime', 'local_arr_datetime', 'Dep Airport Code',
                        'Arr Airport Code', 'Delta', 'rotation', 'remove', 'conn']]

            connections.query("in_conn in @valid_flights or out_conn in @valid_flights", inplace=True)
            connections[['Time series', 'Specific Aircraft Code', 'FLIGHT', 'FLIGHT_CONNECTED',
                         'SLACK', 'DELTA', 'DELTA_CONNECTED']]

            check_time = round((datetime.now() - prev).total_seconds()/60, 2)
            etl_dict['T_CHECK'].append(check_time)

            print(" Creating csv files... ")
            prev = datetime.now()
            rt_flights.to_csv(FLIGHTS_CSV_FILE,
                              sep=",",
                              mode=mode,
                              header=(mode == 'w'),
                              compression=None)

            connections.to_csv(CONNECTIONS_CSV_FILE,
                               sep=",",
                               mode=mode,
                               header=(mode == 'w'),
                               compression=None)

            mode = 'a'
            save_time = round((datetime.now() - prev).total_seconds()/60, 2)
            etl_dict['T_SAVE'].append(save_time)

    df_etl = DataFrame(etl_dict)
    df_etl.to_csv(ETL_CSV_FILE,
                  sep=",",
                  mode='w',
                  header=True,
                  compression=None)


def analize_scenarios(*, filename: str, group_by_week: bool):

    input_flights = read_flights_from_csv(filename=filename)

    group_by_cols = [Grouper(key='Time series', freq='W'), 'Specific Aircraft Code'] if group_by_week else \
        ['Time series', 'Specific Aircraft Code']

    # Creating a DataFrame with the minimum and maximum number of unscheduled flights per week
    dfc1 = input_flights.groupby(by=group_by_cols)[
        ['remove_1', 'remove_2', 'remove_3', 'remove_4', 'remove_5']].sum()
    dfc1['MIN_UNSCH'] = dfc1.min(axis=1)
    dfc1['MAX_UNSCH'] = dfc1.max(axis=1)
    dfc1['MEAN_UNSCH'] = dfc1.mean(axis=1).round(decimals=0)

    # Creating a DataFrame with the minimum and maximum number of aircrafts used per week
    dfc2 = input_flights.groupby(by=group_by_cols)[['aircraft_id_1', 'aircraft_id_2_2hr', 'aircraft_id_2_3hr',
                                                    'aircraft_id_2_4hr', 'aircraft_id_2_5hr']].max()
    dfc2['MIN_NAC'] = dfc2.min(axis=1)
    dfc2['MAX_NAC'] = dfc2.max(axis=1)
    dfc2['MEAN_NAC'] = dfc2.mean(axis=1).round(decimals=0)

    # Creating a DataFrame with the number of flights to be operated per week
    dfc3 = input_flights.groupby(by=group_by_cols)['Flight No'].count().reset_index(name='NF')

    # Merging the three DataFrames to include in a single one all KPI columns
    df_final = merge(dfc3, merge(dfc1[['MIN_UNSCH', 'MAX_UNSCH', 'MEAN_UNSCH']],
                                 dfc2[['MIN_NAC', 'MAX_NAC', 'MEAN_NAC']],
                                 on=['Time series', 'Specific Aircraft Code']),
                     on=['Time series', 'Specific Aircraft Code'])

    # Writing the contents of this final DataFrame to the CSV file SCENARIOS_CSV_FILE
    df_final.to_csv(SCENARIOS_CSV_FILE,
                    sep=",",
                    header=True,
                    compression=None)


if __name__ == "__main__":

    if SCENARIO_OP == 0:
        prepare_scenarios(filename=DATA_CSV_FILE,
                          planning_dates=[('03/04/2019 00:00', '03/10/2019 00:00'), #Month, day, year
                                          ('05/20/2019 00:00', '05/26/2019 00:00'),
                                          ('06/10/2019 00:00', '06/16/2019 00:00')],
                          fleet_types=['320'],
                          retimings=RET_VALUES,
                          airlines=AIRLINE)
    else:
        analize_scenarios(filename=DATA_CSV_FILE,
                          group_by_week=group_by_week)
