from numpy import where
from datetime import timedelta
from df_operations import create_dict
from plotly.figure_factory import create_gantt
from project_constants import ROTATIONS_CSV_FILE, ROTS_COLS, GANTTCHART_DAY, GANTTCHART_FLEET,GANTTCHART_SOL_NUM
from pandas import read_csv, to_datetime, Interval
from math import floor


def int2time(*, date, mins):
    h = floor(mins/60)

    h1 = h - 24 if h >= 24 else h

    if h >= 24:
        n = str(int(date[-2:]) + 1)
        if len(n) < 2:
            n = '0' + n
        date = date[:-2] + n

    hourmin = str(h1)

    if len(hourmin) < 2:
        hourmin = '0' + hourmin

    m = mins - h * 60

    hourmin += ':' + str(m)

    if len(hourmin) < 5:
        hourmin = hourmin[0:3] + '0' + hourmin[-1]

    return date + ' ' + hourmin


def draw_rotations(*, rotations, date, fleet, n_sol):
    df = rotations.loc[(rotations.DATE == date) & (rotations.FLEET == fleet) & (rotations.N_SOL == n_sol)]
    if df.empty:
        print('No solution was found for day: {}, fleet: {}, solution: {}'.format(date, fleet, n_sol))
        exit(1)
    
    df['STD'] = df['STD'].astype('datetime64')
    df['STA'] = df['STA'].astype('datetime64')
    df.reset_index(inplace=True)
    df.ROTATION = df.ROTATION.astype(str).astype(int)
    df = df.sort_values(by=['ROTATION'])
    df["Task"] = df["ROTATION"]
    df["Start"] = df["STD"]
    df["Finish"] = df["STA"]
    df['max_rank'] = df['ROTATION'].rank(method='max')
    df["Center"] = df.apply(lambda row: Interval(left=row["STD"], right=row["STA"]).mid, axis=1)
    df['Status'] = where(df.DELTA == 0, 'FLIGHT', 'FLIGHT_RT')

    tasks = {a for a in df["Task"]}
    tasks = sorted(tasks, reverse=True)
    dict_tasks = {list(tasks)[i]: i for i in range(0, len(tasks))}

    colors = {'FLIGHT': 'rgb(70,130,180)',
              'FLIGHT_RT': 'rgb(0,139,139)'}

    task_num = len(tasks)

    fig = create_gantt(df.to_dict('records'),
                       colors=colors,
                       index_col='Status',
                       show_colorbar=True,
                       group_tasks=True,
                       bar_width=0.25)

    date_min = df['STD'].min() - timedelta(minutes=30)
    date_max = df['STA'].max() + timedelta(minutes=30)

    fig['layout'].update(autosize=False,
                         width=1500,
                         height=300 + 100 * task_num,
                         xaxis_range=[date_min, date_max])

    # add annotations
    annot_font_size = 12
    annots = [{"x": row["Center"],
               "y": dict_tasks[row["Task"]],
               "text": row["FLIGHT_NUMBER"],
               "showarrow": False,
               "font": {'size': annot_font_size, 'color': 'yellow'}
               } for index, row in df.iterrows()]

    vertical_text_x_disp = 5
    std_annots = [{"x": row["Start"] - timedelta(minutes=vertical_text_x_disp),
        "y": dict_tasks[row["Task"]],
        "text": str(row["STD"])[-8:-3],
        "showarrow": False,
        "textangle": 90,
        "font": {'size': annot_font_size, 'color': 'green'}
    } for index, row in df.iterrows()]

    sta_annots = [{"x": row["Finish"] + timedelta(minutes=vertical_text_x_disp),
        "y": dict_tasks[row["Task"]],
        "text": str(row["STA"])[-8:-3],
        "showarrow": False,
        "textangle": 90,
        "font": {'size': annot_font_size, 'color': 'green'}
    } for index, row in df.iterrows()]

    dep_annots = [{"x": row["Start"] + timedelta(minutes=vertical_text_x_disp),
        "y": dict_tasks[row["Task"]],
        "text": row["DEP_AIRPORT"],
        "showarrow": False,
        "textangle": 90,
        "font": {'size': annot_font_size, 'color': 'black'}
    } for index, row in df.iterrows()]

    arr_annots = [{"x": row["Finish"] - timedelta(minutes=vertical_text_x_disp),
        "y": dict_tasks[row["Task"]],
        "text": row["ARR_AIRPORT"],
        "showarrow": False,
        "textangle": 90,
        "font": {'size': annot_font_size, 'color': 'black'}
    } for index, row in df.iterrows()]

    df_retimed = df[df.DELTA != 0]
    text_y_disp = task_num / 100
    delta_annots = [{"x": row["Center"],
                     "y": dict_tasks[row["Task"]] - 0.15,
                     "text": row["DELTA"],
                     "showarrow": False,
                     "font": {'size': annot_font_size, 'color': 'brown'}
                     } for index, row in df_retimed.iterrows()]

    # plot figure
    fig['layout']['annotations'] = annots + std_annots + sta_annots + dep_annots + arr_annots + delta_annots

    fig.show()


# DAYS = ['2019-03-09', '2019-05-25', '2019-06-15']
#
if __name__ == "__main__":

    try:
        rots_types_dict = create_dict(ROTS_COLS)
        rotations = read_csv(ROTATIONS_CSV_FILE, sep=",", dtype=rots_types_dict, header=0)
        draw_rotations(rotations=rotations, date=GANTTCHART_DAY, fleet=GANTTCHART_FLEET, n_sol=GANTTCHART_SOL_NUM)

    except Exception as e:
        print(str(e))
        raise ValueError(str(e))
