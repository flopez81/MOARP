from numpy import int8, uint8, int16, int32, uint16, uint32

KPI_COLS_TYPES = {
    'DATE': str,
    'FLEET': str,
    'N_SOL': uint8,
    'IS_DOMINATED': uint8,
    'N_AC': uint16,
    'N_FLIGHTS': uint16,
    'COVERAGE': float,
    'RT_VAL': float,
    'N_RT': float,
    'TOTAL_SLACK': float,
    'N_UNSCH_F': uint16,
    'N_SWAPS': float,
    'ABS_DELAY': float,
    'ACT_HOURS': float
}

COLS_TYPES = {
    'Dep Airport Code': str,
    'local_dep_datetime': str,
    'local_arr_datetime': str,
    'Carrier Code': str,
    'Local Days Of Op': str,
    'Seats': int,
    'Aircraft Range (NM)': int,
    'Aircraft Max': int,
    'Dep IATA Country Name': str,
    'Arr IATA Country Name': str,
    'Aircraft Cruise Speed (MPH)': int,
    'GCD (km)': int,
    'Arr Airport Code': str,
    'Specific Aircraft Code': str,
    'Codeshare Car 1': str,
    'Flight No': str,
    'Mainline/Low Cost': str,
    'Seats (Total)': int,
    'Time series': str,
    'FLIGHT': str,
    'FLIGHT_CONNECTED': str,
    'SLACK': uint16,
    'DELTA': int16,
    'DELTA_CONNECTED': int16,
    'FLEET': str,
    'DATE': str,
    'N_AC': uint16,
    'N_FLIGHTS': int,
    'COVERAGE': float,
    'RT_VAL': float,
    'N_RT': int,
    'TOTAL_SLACK': int,
    'MIN_SLACK': int,
    'MAX_SLACK': int,
    'N_UNSCH_F': int,
    'N_SWAPS': int16,
    'ABS_DELAY': float,
    'ACT_HOURS': float
}


def create_dict(columns):
    column_dict = {key: value for c in columns for key, value in COLS_TYPES.items() if key == c}
    return column_dict


def create_dict_kpi(columns):
    column_dict = {key: value for c in columns for key, value in KPI_COLS_TYPES.items() if key == c}
    return column_dict
