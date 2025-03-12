import datetime
from math import ceil
from pandas import DataFrame, merge, to_timedelta
from datetime import datetime, timedelta
from project_constants import ERROR_CODE, ROT_DICT_KEYS, KPI_DICT_KEYS, ARC_COL, INI_SOL_COL, EPS_OBJS, \
    OBJ_FUN, UNSCH_DICT_KEYS, PER_DICT_KEYS, SAVE_STATUS, MIN_TAT, SWAPS_CSV_FILE, SWAPS_COLS, WRITE_SWAP_FILE


class AircraftRoutingOutputs:
    def __init__(self, fleets):
        self.day = ''
        self.prev_day = ''
        self.fleet = ''
        self.output_save_status = SAVE_STATUS['SAVE_OUTPUT']
        self.temp_KPIs = False
        self.min_slack = 0
        self.max_slack = 0
        self.n_opt_steps = int(0)
        self.n_flights = int(0)
        self.obj_type = int(0)
        self.dict_rot = {key: list() for key in ROT_DICT_KEYS}
        self.dict_rot_temp = self.dict_rot.copy()
        self.dict_KPIs = {key: list() for key in KPI_DICT_KEYS}
        self.dict_KPIs_temp = self.dict_KPIs.copy()
        self.dict_per = {key: list() for key in PER_DICT_KEYS}
        self.dict_unsch_flights = {key: list() for key in UNSCH_DICT_KEYS}
        self.last_tasks = {f: list() for f in fleets}
        self.code_err = []
        self.err_dict = dict()
        self.err_day_dict = dict()
        self.err_codes_statuses = [False] * len(ERROR_CODE)
        self.runtime = timedelta(minutes=0)
        self.mode = 'w'
        self.kpi = {'N_UNSCH_F': 0,
                    'N_AC': 0,
                    'N_RT': 0,
                    'RT_VAL': 0,
                    'MIN_TOTAL_SLACK': 0,
                    'MAX_TOTAL_SLACK': 0,
                    'MIN_SLACK': 0,
                    'MAX_SLACK': 0,
                    'IS_PARETO': 1,
                    'N_SWAPS': 0,
                    'ABS_DELAY': 0,
                    'ACT_HOURS': 0}

    def update_instance(self, params, SORs0):
        self.day = params.day
        self.prev_day = self.day
        self.obj_type = OBJ_FUN['AIRLINE_SOL']
        self.output_save_status = SAVE_STATUS['SAVE_OUTPUT']
        self.min_slack = 0
        self.max_slack = 0
        self.n_opt_steps = int(0)
        self.fleet = params.fleet
        self.n_flights = len(params.tasks)

        # Building the rotations and kpi's related to the solution given by the network department
        slacks = {(a[ARC_COL['FLIGHT']], a[ARC_COL['FLIGHT_CONNECTED']]): a[ARC_COL['SLACK']]
                  for a in params.connections}

        # Ordering data by rotation id first, then by departure time, and so on
        SORs = sorted(SORs0)

        n_rots = 1
        self.ini_current_rots()
        self.order = 1
        n_flights = 0
        n_flights_rot = 0

        id_rot = SORs[0][0]
        act_arcs0 = list()

        prev_flight = SORs[0][INI_SOL_COL['flight']]
        n_flights += 1
        n_flights_rot += 1
        self.update_rots_dict(id_task=prev_flight,
                              dep_airport=SORs[0][INI_SOL_COL['dep_airport']],
                              arr_airport=SORs[0][INI_SOL_COL['arr_airport']],
                              std=SORs[0][INI_SOL_COL['std']],
                              sta=SORs[0][INI_SOL_COL['sta']],
                              rot=n_rots)

        # Considerar el campo conn para saber el siguiente vuelo
        # Si remove = 1, el vuelo del renglón no se mete en la rotación
        for s in SORs[1:]:
            # print(s[INI_SOL_COL['rotation']], s[INI_SOL_COL['flight']], s[INI_SOL_COL['dep_airport']],
            #      s[INI_SOL_COL['arr_airport']], s[INI_SOL_COL['remove']], s[INI_SOL_COL['conn']])

            if id_rot != s[INI_SOL_COL['rotation']]:
                if n_flights_rot:
                    self.order = 1
                    n_rots += 1

                n_flights_rot = 0
                id_rot = s[INI_SOL_COL['rotation']]

                if not s[INI_SOL_COL['remove']]:
                    self.update_rots_dict(id_task=s[INI_SOL_COL['flight']],
                                          dep_airport=s[INI_SOL_COL['dep_airport']],
                                          arr_airport=s[INI_SOL_COL['arr_airport']],
                                          std=s[INI_SOL_COL['std']],
                                          sta=s[INI_SOL_COL['sta']],
                                          rot=n_rots)
                    n_flights += 1
                    n_flights_rot += 1
            elif not s[INI_SOL_COL['remove']]:
                act_arcs0.append((prev_flight, s[INI_SOL_COL['flight']]))
                self.update_rots_dict(id_task=s[INI_SOL_COL['flight']],
                                      dep_airport=s[INI_SOL_COL['dep_airport']],
                                      arr_airport=s[INI_SOL_COL['arr_airport']],
                                      std=s[INI_SOL_COL['std']],
                                      sta=s[INI_SOL_COL['sta']],
                                      rot=n_rots,
                                      arc_slack=slacks[(prev_flight, s[INI_SOL_COL['flight']])])
                n_flights += 1
                n_flights_rot += 1

            prev_flight = s[INI_SOL_COL['flight']]

        self.kpi = {'N_AC': n_rots,
                    'N_UNSCH_F': self.n_flights - n_flights,
                    'RT_VAL': 0,
                    'N_RT': 0,
                    'IS_DOMINATED': 0,
                    'ABS_DELAY': 0,
                    'ACT_HOURS': 0,
                    'N_SWAPS': 0,
                    'TOTAL_SLACK': sum(slacks[a] for a in act_arcs0),
                    'MIN_SLACK': min(slacks[a] for a in act_arcs0),
                    'MAX_SLACK': max(slacks[a] for a in act_arcs0)}

        self.update_kpis_dict()
        self.update_unsch_flights(params=params)
        self.output_save_status = SAVE_STATUS['NO_SAVE']

        self.dict_rot_temp = {key: list() for key in ROT_DICT_KEYS}
        self.dict_KPIs_temp = {key: list() for key in KPI_DICT_KEYS}
        self.dict_per_temp = {key: list() for key in PER_DICT_KEYS}

    def ini_current_rots(self):
        self.current_rots_sol = {'ROTATION': list(),
                                  'ORDER': list(),
                                  'FLIGHT_NUMBER': list(),
                                  'DEP_AP': list(),
                                  'ARR_AP': list(),
                                  'STD': list(),
                                  'STA': list()}

    def update_ilp_output(self, obj_type, n_opt_steps):
        self.obj_type = obj_type
        self.n_opt_steps = n_opt_steps
        self.last_tasks[self.fleet] = list()

    def update_rots_dict(self, id_task, dep_airport, arr_airport, std, sta, rot, arc_slack=0, delta=0):
        std = datetime.strptime(std, '%Y-%m-%d %H:%M:%S')
        #if self.current_rots_sol[rot]['STD']:
        #    self.current_rots_sol[rot]['SLACK'].append((std - (self.current_rots_sol[rot]['STA'][-1]
        #                                        + timedelta(minutes=MIN_TAT))).total_seconds()/60)
        #else:
        #    self.current_rots_sol[rot]['SLACK'].append(0)

        self.current_rots_sol['ROTATION'].append(rot)
        self.current_rots_sol['ORDER'].append(self.order)
        self.current_rots_sol['FLIGHT_NUMBER'].append(id_task)
        self.current_rots_sol['DEP_AP'].append(dep_airport)
        self.current_rots_sol['ARR_AP'].append(arr_airport)
        self.current_rots_sol['STD'].append(std)
        self.current_rots_sol['STA'].append(datetime.strptime(sta, '%Y-%m-%d %H:%M:%S') + timedelta(minutes=MIN_TAT))

        self.order += 1
        if self.output_save_status == SAVE_STATUS['SAVE_OUTPUT']:
            self.dict_rot['DATE'].append(self.day)
            self.dict_rot['FLEET'].append(self.fleet)
            self.dict_rot['UP_SLACK_BOUND'].append(self.max_slack)
            self.dict_rot['N_SOL'].append(self.n_opt_steps)
            self.dict_rot['IS_DOMINATED'].append(1)
            self.dict_rot['OBJECTIVE'].append(self.obj_type)
            self.dict_rot['ROTATION'].append(rot)
            self.dict_rot['FLIGHT_NUMBER'].append(id_task)
            self.dict_rot['DEP_AIRPORT'].append(dep_airport)
            self.dict_rot['ARR_AIRPORT'].append(arr_airport)
            self.dict_rot['STD'].append(std)
            self.dict_rot['STA'].append(sta)
            self.dict_rot['DELTA'].append(delta)
            self.dict_rot['SLACK'].append(arc_slack)
        else:
            self.dict_rot_temp['DATE'].append(self.day)
            self.dict_rot_temp['FLEET'].append(self.fleet)
            self.dict_rot_temp['UP_SLACK_BOUND'].append(self.max_slack)
            self.dict_rot_temp['N_SOL'].append(self.n_opt_steps)
            self.dict_rot_temp['IS_DOMINATED'].append(1)
            self.dict_rot_temp['OBJECTIVE'].append(self.obj_type)
            self.dict_rot_temp['ROTATION'].append(rot)
            self.dict_rot_temp['FLIGHT_NUMBER'].append(id_task)
            self.dict_rot_temp['DEP_AIRPORT'].append(dep_airport)
            self.dict_rot_temp['ARR_AIRPORT'].append(arr_airport)
            self.dict_rot_temp['STD'].append(std)
            self.dict_rot_temp['STA'].append(sta)
            self.dict_rot_temp['DELTA'].append(delta)
            self.dict_rot_temp['SLACK'].append(arc_slack)

    def update_per_dict(self, cpu_time, mdl):
        self.dict_per['DATE'].append(self.day)
        self.dict_per['FLEET'].append(self.fleet)
        self.dict_per['UP_SLACK_BOUND'].append(self.max_slack)
        self.dict_per['N_SOL'].append(self.n_opt_steps)
        self.dict_per['OBJECTIVE'].append(self.obj_type)
        self.dict_per['N_VARS'].append(mdl.variables.get_num())
        self.dict_per['N_I_VARS'].append(mdl.variables.get_num_binary())
        self.dict_per['N_C_VARS'].append(mdl.variables.get_num() - mdl.variables.get_num_binary())
        self.dict_per['N_CONS'].append(mdl.linear_constraints.get_num())
        self.dict_per['N_NZES'].append(mdl.linear_constraints.get_num_nonzeros())
        self.dict_per['STATUS'].append(mdl.solution.status[mdl.solution.get_status()])
        self.dict_per['TIME'].append(cpu_time)

    def update_kpis_dict(self):
        self.calc_kpis2()

        if self.output_save_status == SAVE_STATUS['SAVE_OUTPUT']:
            self.dict_KPIs['DATE'].append(self.day)
            self.dict_KPIs['FLEET'].append(self.fleet)
            self.dict_KPIs['UP_SLACK_BOUND'].append(self.max_slack)
            self.dict_KPIs['N_SOL'].append(self.n_opt_steps)
            self.dict_KPIs['SCORE'].append(0)
            self.dict_KPIs['IS_DOMINATED'].append(self.kpi['IS_DOMINATED'])
            self.dict_KPIs['OBJECTIVE'].append(self.obj_type)
            self.dict_KPIs['N_AC'].append(self.kpi['N_AC'])
            self.dict_KPIs['N_FLIGHTS'].append(self.n_flights - self.kpi['N_UNSCH_F'])
            self.dict_KPIs['COVERAGE'].append(int(100 * ((self.n_flights - self.kpi['N_UNSCH_F']) / self.n_flights)))
            self.dict_KPIs['RT_VAL'].append(self.kpi['RT_VAL'])
            self.dict_KPIs['N_RT'].append(self.kpi['N_RT'])
            self.dict_KPIs['TOTAL_SLACK'].append(self.kpi['TOTAL_SLACK'])
            self.dict_KPIs['MIN_SLACK'].append(self.kpi['MIN_SLACK'])
            self.dict_KPIs['MAX_SLACK'].append(self.kpi['MAX_SLACK'])
            self.dict_KPIs['N_UNSCH_F'].append(self.kpi['N_UNSCH_F'])
            self.dict_KPIs['N_SWAPS'].append(self.kpi['N_SWAPS'])
            self.dict_KPIs['ABS_DELAY'].append(self.kpi['ABS_DELAY'])
            self.dict_KPIs['ACT_HOURS'].append(self.kpi['ACT_HOURS'])
        else:
            self.dict_KPIs_temp['DATE'].append(self.day)
            self.dict_KPIs_temp['FLEET'].append(self.fleet)
            self.dict_KPIs_temp['UP_SLACK_BOUND'].append(self.max_slack)
            self.dict_KPIs_temp['N_SOL'].append(self.n_opt_steps)
            self.dict_KPIs_temp['SCORE'].append(0)
            self.dict_KPIs_temp['IS_DOMINATED'].append(0)
            self.dict_KPIs_temp['OBJECTIVE'].append(self.obj_type)
            self.dict_KPIs_temp['N_AC'].append(self.kpi['N_AC'])
            self.dict_KPIs_temp['N_FLIGHTS'].append(self.n_flights - self.kpi['N_UNSCH_F'])
            self.dict_KPIs_temp['COVERAGE'].append(int(100 * ((self.n_flights - self.kpi['N_UNSCH_F']) / self.n_flights)))
            self.dict_KPIs_temp['RT_VAL'].append(self.kpi['RT_VAL'])
            self.dict_KPIs_temp['N_RT'].append(self.kpi['N_RT'])
            self.dict_KPIs_temp['TOTAL_SLACK'].append(self.kpi['TOTAL_SLACK'])
            self.dict_KPIs_temp['MIN_SLACK'].append(self.kpi['MIN_SLACK'])
            self.dict_KPIs_temp['MAX_SLACK'].append(self.kpi['MAX_SLACK'])
            self.dict_KPIs_temp['N_UNSCH_F'].append(self.kpi['N_UNSCH_F'])
            self.dict_KPIs_temp['N_SWAPS'].append(self.kpi['N_SWAPS'])
            self.dict_KPIs_temp['ABS_DELAY'].append(self.kpi['ABS_DELAY'])
            self.dict_KPIs_temp['ACT_HOURS'].append(self.kpi['ACT_HOURS'])

    def calc_kpis1(self, params):
        self.calc_n_unsch_f()
        self.calc_n_used_ac(params=params)
        self.calc_n_rt(params=params)
        self.calc_rt_value(params=params)
        self.calc_slack(params=params)
        self.calc_max_slack(params=params)
        self.calc_min_slack(params=params)

    def calc_kpis2(self):

        # Create a dataframe from the current solution rotations
        df = DataFrame(self.current_rots_sol)

        # Compute the total rotations duration
        dfc = df.copy()
        dfc['MIN_ORDER'] = dfc.groupby(by=['ROTATION'])['ORDER'].transform('min')
        dfc['MAX_ORDER'] = dfc.groupby(by=['ROTATION'])['ORDER'].transform('max')
        
        dfc['STDR'] = 0
        dfc.loc[dfc['ORDER'] == dfc['MIN_ORDER'], 'STDR'] = (dfc['STD'].dt.hour*60 + dfc['STD'].dt.minute)/60
        
        dfc['STAR'] = 0
        dfc.loc[dfc['ORDER'] == dfc['MAX_ORDER'], 'STAR'] = (dfc['STA'].dt.hour*60 + dfc['STA'].dt.minute)/60
        dfc = dfc.loc[(dfc['STDR'] > 0) | (dfc['STAR'] > 0)]

        self.kpi['ACT_HOURS'] = round((dfc['STAR'] - dfc['STDR']).sum(), 2)

        # Compute the total number of swaps
        # Create all possible pairs of flights
        df['Dummy'] = 1
        df1 = merge(df, df, on='Dummy')

        # For determining the number of swaps, we need to filter the pairs of flights that verify three conditions:
        # 1. Each flight belongs to a different rotation
        # 2. The arrival airport of the incoming flight is the same as the departure airport of the outgoing flight
        # 3. The departure time of the outgoing flight is equal or greater than the arrival time + tatmin of
        #    the incoming flight
        df_swaps = df1.loc[df1['ROTATION_x'] != df1['ROTATION_y']].loc[df1['DEP_AP_y'] == df1['ARR_AP_x']].
        loc[df1['STD_y'] >= df1['STA_x']]

        if WRITE_SWAP_FILE:
            df_swaps.to_csv(SWAPS_CSV_FILE, sep=",", header=True, compression=None, mode=self.mode, columns=SWAPS_COLS)
            self.mode = 'a'

        # The number of swaps is the number of pairs of flights satisfying the above conditions
        self.kpi['N_SWAPS'] = df_swaps.shape[0]

        # For comptuing the absorved delay, we need to find first the time gaps between connectable pairs of flights.
        # To this end:
        # 1. The arrival airport of the incoming flight is the same as the departure airport of the outgoing flight
        # 2. The departure time of the outgoing flight is equal or greater than the arrival time + tatmin of
        #    the incoming flight
        df_abs_delay = df1.loc[df1['DEP_AP_y'] == df1['ARR_AP_x']].loc[df1['STD_y'] >= df1['STA_x']]

        # The total absorbed delay is the difference between the departure time of the outgoing flight and
        # the arrival time + tat_min of the incoming flight
        self.kpi['ABS_DELAY'] = (df_abs_delay['STD_y'] - df_abs_delay['STA_x']).sum().total_seconds()/3600 \
            if self.kpi['N_SWAPS'] > 0 else 0

    def calc_n_unsch_f(self):
        self.kpi['N_UNSCH_F'] = sum(self.act_y)

    def calc_n_used_ac(self, params):
        self.kpi['N_AC'] = sum(1 for a in self.act_arcs if params.is_origin(a[0]))

    def calc_n_rt(self, params):
        self.kpi['N_RT'] = ceil(sum(1 for a in self.act_arcs if params.arc_rt_st[a])/2)

    def calc_rt_value(self, params):
        self.kpi['RT_VAL'] = sum(params.get_arc_delta_value(a[1]) for a in self.act_arcs if params.arc_rt_st[a])

    def calc_slack(self, params):
        self.kpi['TOTAL_SLACK'] = sum(params.slacks[a] for a in self.act_arcs)

        if self.obj_type == OBJ_FUN['MIN_TOTAL_SLACK']:
            self.kpi['MIN_TOTAL_SLACK'] = self.kpi['TOTAL_SLACK']
        elif self.obj_type == OBJ_FUN['MAX_TOTAL_SLACK']:
            self.kpi['MAX_TOTAL_SLACK'] = self.kpi['TOTAL_SLACK']

    def calc_max_slack(self, params):
        self.kpi['MAX_SLACK'] = max([params.slacks[a] for a in self.act_arcs])

    def calc_min_slack(self, params):
        self.kpi['MIN_SLACK'] = min([params.slacks[a] for a in self.act_arcs])

    def update_unsch_flights(self, params):
        n_disconnected_tasks = len(params.disconnected_tasks)
        self.dict_unsch_flights['DATE'].extend([self.day] * n_disconnected_tasks)
        self.dict_unsch_flights['FLEET'].extend([self.fleet] * n_disconnected_tasks)
        self.dict_unsch_flights['FLIGHT_NUMBER'].extend(params.disconnected_tasks)
        # self.dict_unsch_flights['DEP_AIRPORT'].append(dep_airport)
        # self.dict_unsch_flights['ARR_AIRPORT'].append(arr_airport)
        # self.dict_unsch_flights['STD'].append(std)
        # self.dict_unsch_flights['STA'].append(sta)

    def get_dict_per(self):
        return self.dict_per

    def get_dict_kpis(self):
        return self.dict_KPIs

    def get_dict_rot(self):
        return self.dict_rot

    def no_feasible_solution(self):
        return not self.act_arcs

    def update_err_day_dict(self):
        for i, j in enumerate(self.code_err):
            self.err_codes_statuses[j] = True

        if self.day == self.prev_day or self.prev_day == '':
            self.err_day_dict.update({self.fleet: self.err_codes_statuses})
        else:
            self.err_dict.update({self.day: self.err_day_dict})
            self.err_day_dict = {self.fleet: self.err_codes_statuses}

    def update_err_dict(self):
        self.err_dict.update({self.day: self.err_day_dict})
        # self.err_dict.update({(self.day, fleet): self.err_day_dict[fleet] for fleet in self.err_day_dict})

    def get_dict_err(self):
        return self.err_dict

    def find_indices(self, item_to_find):
        return [idx for idx, value in enumerate(self.dict_rot_temp['N_SOL']) if value == item_to_find]
