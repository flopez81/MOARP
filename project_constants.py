# Project constants added & used by developer
SCENARIO_OP = 0      # 0: Prepare data, 1: Analyze data
group_by_week = False  # True: group by week, False: group by day

CONNEC_METHOD = 1     # 0: For loops, 1: DataFrames

EXE_MODE = 0          # 0/1: 0-Manual mode, 1-Database

# Parameters for prepare_data.py
# delta = 5 min (default)
# upper = -15 (default)
# lower = 15 (default)
RET_VALUES = [-15, -10, -5, 0, 5, 10, 15]

# Couples of days to optimize
PLAN_HOR = [('06/03/2019 00:00', '06/09/2019 00:00')]

FLIGHTS_CSV_FILE = 'flights.csv'
CONNECTIONS_CSV_FILE = 'connections.csv'

#PLAN_FLEET = ['319', '320', '321', '32A']
PLAN_FLEET = ['320']

# AK: Air Asia
AIRLINE = ['AK']

# Minimum turn-around-time
MIN_TAT = 30
# Must be changed to a dictionary mapping each airport to a specific TAT, ranging from 30 to 60 min
# For example: MIN_TAT = {'JHB': 35, 'BKI': 30, 'KUL': 40, 'KCH': 45, 'PEN': 50}

####################################################################################

FLEET_2_BASES = {'320': ['JHB', 'BKI', 'KUL', 'KCH', 'PEN'],
                 '333': ['JHB', 'BKI', 'KUL', 'KCH', 'PEN']}

BASE_CAP = {'320': {'JHB': 100, 'BKI': 100, 'KUL': 100, 'KCH': 100, 'PEN': 100},
            '333': {'JHB': 100, 'BKI': 100, 'KUL': 100, 'KCH': 100, 'PEN': 100}
            }
STEP = 0.05

GANTTCHART_DAY = "2019-01-01 00:00:00"
GANTTCHART_FLEET = '320'
GANTTCHART_SOL_NUM = 1
# Definition of file names and their locations

# Input file for ETL (prepare_data.py)
DATA_CSV_FILE = 'AK_Flights2019_color_codes_routing.csv'
ETL_CSV_FILE = 'etl.csv'
SCENARIOS_CSV_FILE = 'scenarios.csv'
PERFORMANCE_CSV_FILE = 'perf.csv'
ROTATIONS_CSV_FILE = 'rots.csv'
KPIS_CSV_FILE = 'kpis.csv'
SWAPS_CSV_FILE = 'swaps.csv'
SWAPS_COLS = ['FLIGHT_NUMBER_x', 'ROTATION_x', 'ARR_AP_x', 'STA_x',
              'FLIGHT_NUMBER_y', 'ROTATION_y', 'DEP_AP_y', 'STD_y']
WRITE_SWAP_FILE = False
ERR_CSV_FILE = 'err.csv'
UNSCHEDULED_FLIGHTS_FILE = 'unscheduled_flights.csv'

REMOVE_COLS = ['remove_1', 'remove_2', 'remove_3', 'remove_4', 'remove_5']

METHOD_COLS = {0: ['aircraft_id_1', 'remove_1', 'conn_1'],
               1: ['aircraft_id_2_2hr', 'remove_2', 'conn_2_2hr'],
               2: ['aircraft_id_2_3hr', 'remove_3', 'conn_2_3hr'],
               3: ['aircraft_id_2_4hr', 'remove_4', 'conn_2_4hr'],
               4: ['aircraft_id_2_5hr', 'remove_5', 'conn_2_5hr']}

#METHOD_COLS = {0: ['aircraft_id_1', 'remove_1', 'conn_1'],
#               1: ['aircraft_id_2_2hr', 'remove_2', 'conn_2_2hr'],
#               2: ['aircraft_id_3_3hr', 'remove_3', 'conn_3_3hr'],
#               3: ['aircraft_id_4_4hr', 'remove_4', 'conn_4_4hr'],
#               4: ['aircraft_id_5_5hr', 'remove_5', 'conn_5_5hr']}

FLIGHTS_COLUMNS = ['Time series', 'Dep Airport Code', 'Carrier Code', 'local_dep_datetime',
                   'local_arr_datetime',
                   'Arr Airport Code', 'Specific Aircraft Code', 'Flight No',
                   'aircraft_id_1', 'remove_1', 'conn_1',
                   'aircraft_id_2_2hr', 'remove_2', 'conn_2_2hr',
                   'aircraft_id_2_3hr', 'remove_3', 'conn_2_3hr',
                   'aircraft_id_2_4hr', 'remove_4', 'conn_2_4hr',
                   'aircraft_id_2_5hr', 'remove_5', 'conn_2_5hr']

FLIGHT_COLS_4_CONN = ['Time series', 'Specific Aircraft Code', 'Flight No',
                      'local_dep_datetime', 'local_arr_datetime',
                      'Dep Airport Code', 'Arr Airport Code', 'Delta']

#FLIGHTS_COLUMNS = ['Time series', 'Dep Airport Code', 'Carrier Code', 'local_dep_datetime', 'local_arr_datetime',
#                   'Arr Airport Code', 'Specific Aircraft Code', 'Flight No',
#                   'aircraft_id_1', 'remove_1', 'conn_1', 'aircraft_id_2_2hr', 'remove_2', 'conn_2_2hr',
#                   'aircraft_id_3_3hr', 'remove_3', 'conn_3_3hr', 'aircraft_id_4_4hr', 'remove_4', 'conn_4_4hr',
#                   'aircraft_id_5_5hr', 'remove_5', 'conn_5_5hr']

DICT_FLIGHTS_COLUMNS = ['Time series', 'Dep Airport Code', 'Carrier Code', 'local_dep_datetime', 'local_arr_datetime',
                        'Arr Airport Code', 'Specific Aircraft Code', 'Flight No', 'rotation', 'remove', 'conn']

FLIGHTS_COLUMNS3 = ['Time series', 'Dep Airport Code', 'Carrier Code', 'local_dep_datetime', 'local_arr_datetime',
                    'Arr Airport Code', 'Specific Aircraft Code', 'Flight No']

PREPARE_FLIGHTS_COLS = ['Time series', 'Specific Aircraft Code', 'Flight No', 'local_dep_datetime',
                        'local_arr_datetime', 'Dep Airport Code', 'Arr Airport Code', 'Delta', 'rotation', 'remove',
                        'conn']

FILLED_FLIGHTS_COLS = ['Time series', 'Specific Aircraft Code', 'Flight No', 'local_dep_datetime',
                       'local_arr_datetime', 'Dep Airport Code', 'Arr Airport Code', 'rotation', 'remove', 'conn']

DATE_COLS = ['local_dep_datetime', 'local_arr_datetime']

ETL_FLIGHT_COL = {'DATE': 0,
                  'FLEET': 1,
                  'FLIGHT': 2,
                  'DEP_TIME': 3,
                  'ARR_TIME': 4,
                  'DEP_AIRPORT': 5,
                  'ARR_AIRPORT': 6,
                  'DELTA': 7}

KPIS_COLUMNS = ['DATE', 'FLEET', 'N_SOL', 'IS_DOMINATED', 'RT_VAL', 'N_RT',
                'TOTAL_SLACK', 'N_SWAPS', 'ABS_DELAY', 'ACT_HOURS']

# Parameters for optimize.py
RET_STATUS = True   # If retiming is considered or not

# Mapping Objective functions names to their identifiers
OBJ_FUN = {'AIRLINE_SOL': 0,
           'MIN_N_UNSCH_F': 1,
           'MIN_N_AC': 2,
           'MIN_TOTAL_SLACK': 3,
           'MAX_TOTAL_SLACK': 4,
           'MIN_N_RT': 5}

OBJ_2_KPI = {1: 'n_unsch_f', 2: 'n_used_ac', 3: 'TOTAL_SLACK', 4: 'TOTAL_SLACK', 5: 'N_RT'}

# Parameters for epsilon_constraint_method
# Defining the number of values to test for each non-primary objective
EPS_KS = {'MIN_TOTAL_SLACK': 1, 'MAX_TOTAL_SLACK': 1}
EPS_OBJS = ['MIN_TOTAL_SLACK', 'MAX_TOTAL_SLACK']
DOM_OBJS = ['N_RT', 'RT_VAL', 'TOTAL_SLACK']
DOM_SIGNS = {'N_RT': 1, 'RT_VAL': 1, 'TOTAL_SLACK': 1}
#DOM_OBJS = ['N_RT', 'RT_VAL', 'TOTAL_SLACK', 'N_SWAPS', 'ABS_DELAY', 'ACT_HOURS']
#DOM_SIGNS = {'N_RT': 1, 'RT_VAL': 1, 'TOTAL_SLACK': 1, 'N_SWAPS': -1, 'ABS_DELAY': -1, 'ACT_HOURS': 1}

# Parameters for other python files
RT_FLIGHTS_COLUMNS = ['Delta', 'Dep Airport Code', 'Arr Airport Code', 'local_dep_datetime', 'local_arr_datetime',
                      'Specific Aircraft Code', 'Flight No', 'Time series']

FLIGHTS_COLS_FROM_INI_SOL = ['rotation', 'local_dep_datetime', 'local_arr_datetime', 'Flight No', 
                             'Dep Airport Code', 'Arr Airport Code', 'remove', 'conn']

INI_SOL_COL = {'rotation': 0,
               'remove': 6,
               'conn': 7,
               'std': 1,
               'sta': 2,
               'flight': 3,
               'dep_airport': 4,
               'arr_airport': 5}

# Cambiar el primer Dep Airport Code por Airport Base
FLIGHTS_COLS_TO_OPT = ["Flight No", "Delta", "Dep Airport Code", "Dep Airport Code", "Arr Airport Code", 
                       "local_dep_datetime", "local_arr_datetime"]

TASK_COL = {'TASK_ID': 0,
            'DELTA': 1,
            'CD_AIRPORT_BASE': 2,
            'CD_AIRPORT_DEP': 3,
            'CD_AIRPORT_ARR': 4,
            'TS_STD_UTC': 5,
            'TS_STA_UTC': 6}

DATETIME_COLUMNS = ['Time series']

ARC_COLUMNS = ['Time series', 'Specific Aircraft Code', 'FLIGHT', 'FLIGHT_CONNECTED', 'SLACK', 'DELTA',
               'DELTA_CONNECTED']

ARCS_COLS_TO_OPT = ["FLIGHT", "FLIGHT_CONNECTED", "SLACK", "DELTA", "DELTA_CONNECTED"]

ARC_COL = {'FLIGHT': 0,
           'FLIGHT_CONNECTED': 1,
           'SLACK': 2,
           'DELTA': 3,
           'DELTA_CONNECTED': 4}

GROUPBY_COLS_TO_OPT = ["Time series", "Specific Aircraft Code"]

INDEX = {'date': 0,
         'fleet': 1,
         'tasks': 2,
         'arcs': 2}

# Constants definition for the warnings and errors codes
ERROR_CODE = {'N_UNSCH_INFEAS': -1,
              'N_AC_INFEAS': 0,
              'RT_CH_INFEAS': 1,
              'RT_VAL_INFEAS': 2,
              'RT_NO_IMPROV': 3,
              'MIN_PREV_INFEAS': 4,
              'NONE_INFEAS': 5,
              'MIN_TOTAL_INFEAS': 6,
              'MAX_TOTAL_INFEAS': 7}

OBJ2ACK_MSG = {OBJ_FUN['MIN_TOTAL_SLACK']: 'Searching Alternative OSRs by minimizing the total slack',
               OBJ_FUN['MAX_TOTAL_SLACK']: 'Searching Alternative OSRs by maximizing the total slack'}

OBJ2ERR_MSG = {OBJ_FUN['MIN_TOTAL_SLACK']: 'Fail to minimize total slack...',
               OBJ_FUN['MAX_TOTAL_SLACK']: 'Fail to maximize total slack...'}

OBJ2ERR_CODE = {OBJ_FUN['MIN_TOTAL_SLACK']: ERROR_CODE['MIN_TOTAL_INFEAS'],
                OBJ_FUN['MAX_TOTAL_SLACK']: ERROR_CODE['MAX_TOTAL_INFEAS']}

ETL_DICT_KEYS = ['WEEK', 'DATE', 'FLEET', 'N_F_RETS', 'T_F_RETS', 'N_CONN', 'T_CONN', 'T_CHECK']

ROT_DICT_KEYS = ['DATE', 'FLEET', 'UP_SLACK_BOUND', 'N_SOL', 'OBJECTIVE', 'IS_DOMINATED', 'ROTATION',
                 'FLIGHT_NUMBER', 'DEP_AIRPORT', 'ARR_AIRPORT', 'STD', 'STA', 'DELTA', 'SLACK']

KPI_DICT_KEYS = ['DATE', 'FLEET', 'UP_SLACK_BOUND', 'N_SOL', 'SCORE', 'OBJECTIVE', 'IS_DOMINATED',
                 'N_AC', 'N_FLIGHTS', 'COVERAGE', 'RT_VAL', 'N_RT', 'TOTAL_SLACK', 'MIN_SLACK', 'MAX_SLACK',
                 'N_UNSCH_F', 'N_SWAPS', 'ABS_DELAY', 'ACT_HOURS']

PER_DICT_KEYS = ['DATE', 'FLEET', 'UP_SLACK_BOUND', 'N_SOL', 'OBJECTIVE', 'N_VARS', 'N_I_VARS', 'N_C_VARS',
                 'N_CONS', 'N_NZES', 'STATUS', 'TIME']

UNSCH_DICT_KEYS = ['DATE', 'FLEET', 'FLIGHT_NUMBER']          #, 'DEP_AIRPORT', 'ARR_AIRPORT', 'STD', 'STA']

SAVE_STATUS = {'NO_SAVE': 0, 'BACKUP': 1, 'SAVE_OUTPUT': 2}

# CPLEX tuning parameters
CPX_TIME_LIMIT = 3600  # CPLEX time limit in seconds
CPX_PARALLEL_MODE = 1  # Possible values:
#  0: CPLEX decides whether to use deterministic or opportunistic search
#  1: Enabling parallel deterministic search
# -1: Enabling parallel opportunistic search

# Maximum number of CPUs that CPLEX can be used to bind threads
# CPX_NCPUs = "00000001"  # 1
# CPX_NCPUs = "00000003"  # 2
CPX_NCPUs = "0000000F"  # 4
# CPX_NCPUs = "000000FF"  # 8
# CPX_NCPUs = "FFFFFFFF" #
# Value must be set in hexadecimal string within "00000001" and "FFFFFFFF"

# Parameters for ganttchart.py
ROTS_COLS = ['FLEET']
