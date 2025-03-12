from numpy import arange
from df_operations import create_dict
from group_data import AircraftRoutingParams
from group_output import AircraftRoutingOutputs
from pandas import read_csv, DataFrame, merge
from model import update_objective, solve_ilp, set_max_total_slack_limitation
from methods import min_n_unsch_flights, min_n_aircrafts, min_rt_changes, min_total_slack, max_total_slack
from project_constants import FLIGHTS_COLS_TO_OPT, ARCS_COLS_TO_OPT, GROUPBY_COLS_TO_OPT, INDEX, \
    RT_FLIGHTS_COLUMNS, DATETIME_COLUMNS, ARC_COLUMNS, FLIGHTS_COLS_FROM_INI_SOL, \
    FLIGHTS_CSV_FILE, CONNECTIONS_CSV_FILE, KPIS_CSV_FILE, ROTATIONS_CSV_FILE, PERFORMANCE_CSV_FILE, \
    ERR_CSV_FILE, UNSCHEDULED_FLIGHTS_FILE, OBJ_FUN, SAVE_STATUS, STEP


def read_data_from_csv():
    print('Read flights data')
    flights_types_dict = create_dict(RT_FLIGHTS_COLUMNS)
    flights = read_csv(FLIGHTS_CSV_FILE,
                       sep=",",
                       dtype=flights_types_dict,
                       header=0,
                       parse_dates=DATETIME_COLUMNS)

    # flights['Flight No'] = flights['Flight No'].astype('str')

    print('Read connections data')
    arcs_types_dict = create_dict(ARC_COLUMNS)
    connections = read_csv(CONNECTIONS_CSV_FILE,
                           sep=",",
                           dtype=arcs_types_dict,
                           header=0,
                           parse_dates=DATETIME_COLUMNS)

    return flights, connections


def aircraft_routing(*, flights: DataFrame, arcs: DataFrame):
    print('Preparing Scenarios for optimization')

    # Grouping tasks by dt_rotation and base_equipment
    no_rt_flights = flights.loc[flights['Delta'] == 0].copy()
    no_rt_flights["INISOL"] = no_rt_flights[FLIGHTS_COLS_FROM_INI_SOL].apply(tuple, axis='columns')
    ini_sol = no_rt_flights.groupby(GROUPBY_COLS_TO_OPT)["INISOL"].apply(lambda x: x.unique()).reset_index()

    flights["RT_FLIGHTS"] = flights[FLIGHTS_COLS_TO_OPT].apply(tuple, axis='columns')
    rt_flights = flights.groupby(GROUPBY_COLS_TO_OPT)["RT_FLIGHTS"].apply(lambda x: x.unique()).reset_index()

    # Creating a new column labeled as CONNECTIONS in links dataframe with a tuple containing specific and grouping
    #  by dt_rotation and base_equipment
    arcs["CONNECTIONS"] = arcs[ARCS_COLS_TO_OPT].apply(tuple, axis='columns')
    grouped_links = arcs.groupby(GROUPBY_COLS_TO_OPT)["CONNECTIONS"].apply(lambda x: x.unique()).reset_index()

    if len(rt_flights.index) != len(grouped_links.index):
        print('Number of rows of dataframes ext_tasks and grouped_links are different')
        df = merge(rt_flights, grouped_links, on=GROUPBY_COLS_TO_OPT, how='left', indicator='Exist')
        missed_rows = df.loc[df.Exist != 'both']
        print('missed rows: {}'.format(missed_rows))
        exit(-1)

    outputs = AircraftRoutingOutputs(fleets=rt_flights['Specific Aircraft Code'].unique())

    # Determining the optimal rotations for each date and fleet
    print('Start optimization process')
    for i, row in grouped_links.iterrows():
        print('Day {}, Fleet {}'.format(row[INDEX['date']], row[INDEX['fleet']]))

        # Creating params object using selected row data from grouped_links dataframe
        params = AircraftRoutingParams(tasks=rt_flights.iloc[i, INDEX['tasks']],
                                       connections=row[INDEX['arcs']],
                                       day=row[INDEX['date']],
                                       fleet=row[INDEX['fleet']])

        # Updating output data according to selected combination day-equipment type
        outputs.update_instance(params=params, SORs0=tuple(ini_sol.iloc[i, INDEX['tasks']]))

        if not params.are_there_tasks():
            continue

        if epsilon_constraint_method(params=params, outputs=outputs) == -1:
            continue

    outputs.update_err_dict()

    print('Save models performance...')
    df_ilp_per = DataFrame(outputs.get_dict_per())
    df_ilp_per.to_csv(PERFORMANCE_CSV_FILE, sep=",", header=True, compression=None)

    print('Save constructed rotations...')
    df_ilp_output = DataFrame(outputs.get_dict_rot())
    df_ilp_output.to_csv(ROTATIONS_CSV_FILE, sep=",", header=True, compression=None)

    print('Save calculated KPIs...')
    df_ilp_kpis = DataFrame(outputs.get_dict_kpis())
    df_ilp_kpis.to_csv(KPIS_CSV_FILE, sep=",", header=True, compression=None)

    print('Save unscheduled flights...')
    df_ilp_kpis = DataFrame(outputs.dict_unsch_flights)
    df_ilp_kpis.to_csv(UNSCHEDULED_FLIGHTS_FILE, sep=",", header=True, compression=None)

    print('Save errors...')
    df_err = DataFrame(outputs.get_dict_err())
    df_err.to_csv(ERR_CSV_FILE, sep=",", header=True, compression=None)


def epsilon_constraint_method(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs) -> int:
    print('Running Epsilon Constraint Method...')

    # All optimizations can be done in parallel in Step 1 except for the two first
    print('Step 1. Optimizing all objectives independently')

    # Minimizing the number of unscheduled flights
    mdl = min_n_unsch_flights(params=params, outputs=outputs)
    if outputs.no_feasible_solution():
        outputs.update_err_day_dict()
        return -1

    # Minimizing the number of aircraft
    mdl = min_n_aircrafts(params=params, mdl=mdl, outputs=outputs)
    if outputs.no_feasible_solution():
        outputs.update_err_day_dict()
        return -1
    min_slack = outputs.kpi['TOTAL_SLACK']
    print(' Current slack: {}, Min_slack {}'.format(outputs.kpi['TOTAL_SLACK'], min_slack))

    # Minimizing the total slack
    min_total_slack(params=params, mdl=mdl, outputs=outputs)
    if outputs.no_feasible_solution():
        outputs.update_err_day_dict()
        return -1
    min_slack = min([min_slack, outputs.kpi['TOTAL_SLACK']])
    print(' Current slack: {}, Min_slack {}'.format(outputs.kpi['TOTAL_SLACK'], min_slack))

    # Maximizing the total slack
    max_total_slack(params=params, mdl=mdl, outputs=outputs)
    if outputs.no_feasible_solution():
        outputs.update_err_day_dict()
        return -1
    max_slack = outputs.kpi['TOTAL_SLACK']
    print(' Max slack: {}'.format(outputs.kpi['TOTAL_SLACK']))

    print('Step 2. Generate a multi-dimensional grid structure with objective function ranges')
    alphas = list(arange(0, 1 + STEP, STEP))
    outputs.output_save_status = SAVE_STATUS['BACKUP']
    params.set_obj_type(OBJ_FUN['MIN_N_RT'])
    update_objective(params=params, mdl=mdl)

    # Limiting the maximum total slack
    params.rhs = list()
    params.senses = list()
    params.lin_expr = list()
    set_max_total_slack_limitation(params=params, outputs=outputs)
    mdl.linear_constraints.add(lin_expr=params.lin_expr, senses=params.senses, rhs=params.rhs)

    params.n_opt_steps = 0
    n_const = mdl.linear_constraints.get_num()
    for alpha in alphas:
        print('alpha: ', alpha)
        outputs.max_slack = min_slack + alpha*(max_slack - min_slack)
        print(' max_slack: ', outputs.max_slack)

        # Updating the rhs term related to slack boundaries
        mdl.linear_constraints.set_rhs([(n_const - 1, outputs.max_slack)])

        # P_k: Minimize number of re-timings subject to max_slack
        solve_ilp(mdl=mdl, params=params, outputs=outputs)
        if outputs.no_feasible_solution():
            outputs.update_err_day_dict()
            continue

        print(' Solution: ',
              'N_RT = ', outputs.kpi['N_RT'], ', ',
              'RT_VAL = ', outputs.kpi['RT_VAL'], ', ',
              'TOTAL_SLACK = ', outputs.kpi['TOTAL_SLACK'], ', ',
              'ACT_HOURS = ', outputs.kpi['ACT_HOURS'], ', ',
              'N_SWAPS = ', outputs.kpi['N_SWAPS'], ', ',
              'ABS_DELAY = ', outputs.kpi['ABS_DELAY'])

    sols = list(range(len(outputs.dict_KPIs_temp['DATE'])))

    print('Step 3. Characterize the {} solutions found'.format(len(sols)))
    print(' Number of solutions temporary saved {}'.format(params.n_opt_steps))

    for s in sols:
        print(' Solution {}: n_rt={}, slack={}, n_swaps={}, abs_delay={}, act_hours={}'.
              format(s + 1, outputs.dict_KPIs_temp['N_RT'][s], outputs.dict_KPIs_temp['TOTAL_SLACK'][s],
                     outputs.dict_KPIs_temp['N_SWAPS'][s], outputs.dict_KPIs_temp['ABS_DELAY'][s],
                     outputs.dict_KPIs_temp['ACT_HOURS'][s]))

        sols1 = sols.copy()
        sols1.remove(s)
        for s1 in sols1:
            if outputs.dict_KPIs_temp['N_RT'][s1] <= outputs.dict_KPIs_temp['N_RT'][s] and \
                    outputs.dict_KPIs_temp['TOTAL_SLACK'][s1] <= outputs.dict_KPIs_temp['TOTAL_SLACK'][s]:

                print(' Solution {} is dominated by solution {}: n_rt={}, slack={}'.
                      format(s + 1, s1 + 1, outputs.dict_KPIs_temp['N_RT'][s1],
                             outputs.dict_KPIs_temp['TOTAL_SLACK'][s1],
                             outputs.dict_KPIs_temp['N_SWAPS'][s1]))

                outputs.dict_KPIs_temp['IS_DOMINATED'][s] = 1
                break

        # Adding the solution to the rotations and KPIs dictionaries
        ind = outputs.find_indices(s + 1)

        if ind:
            for el in outputs.dict_rot:
                outputs.dict_rot[el].extend(outputs.dict_rot_temp[el][ind[0]:ind[-1] + 1])
        else:
            print(' Temporary Rotation data was not found!')
            exit(-1)

        for kpi in outputs.dict_KPIs_temp:
            outputs.dict_KPIs[kpi].append(outputs.dict_KPIs_temp[kpi][s])

    return 0


if __name__ == "__main__":

    try:
        flights, connections = read_data_from_csv()
        aircraft_routing(flights=flights, arcs=connections)

    except Exception as e:
        print(str(e))
        raise ValueError(str(e))
