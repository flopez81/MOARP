from project_constants import TASK_COL, ARC_COL, OBJ_FUN, FLEET_2_BASES


class AircraftRoutingParams:
    def __init__(self, tasks, connections, fleet, day):
        self.day = day
        self.fleet = fleet
        self.bases = FLEET_2_BASES[fleet]
        self.connections = connections
        self.n_used_ac_bck = False
        self.first_opt_goal = True
        self.set_kpi_limit = True

        self.tasks2types = {'O': 'SOURCE', 'D': 'SINK'}
        self.tasks = sorted(list(set([t[TASK_COL['TASK_ID']] for t in tasks])))
        self.tasks2types.update({t[TASK_COL['TASK_ID']]: 'F' for t in tasks})

        self.deltas_in_tasks = dict()
        for t in tasks:
            self.deltas_in_tasks.setdefault(t[TASK_COL['TASK_ID']], []).append(t[TASK_COL['DELTA']])

        deltas = set([t[TASK_COL['DELTA']] for t in tasks])
        self.delta2pos = {j: d for d, j in enumerate(deltas)}
        self.pos2delta = {d: j for d, j in enumerate(deltas)}

        self.tasks_airports = {(t[TASK_COL['TASK_ID']] + '_' + str(self.delta2pos[t[TASK_COL['DELTA']]])):
                                   (t[TASK_COL['CD_AIRPORT_DEP']], t[TASK_COL['CD_AIRPORT_ARR']],
                                    t[TASK_COL['TS_STD_UTC']], t[TASK_COL['TS_STA_UTC']]) for t in tasks}

        self.kpi_limit_stat = {'n_unsch_f': False,
                               'n_used_ac': False,
                               'N_RT': False,
                               'RT_VAL': False,
                               'tat': False,
                               'TOTAL_SLACK': False,
                               'MIN_SLACK': False,
                               'MAX_SLACK': False}

        self.n_opt_steps = 0
        self.obj_type = OBJ_FUN['AIRLINE_SOL']
        self.origins = ['O']
        self.destinations = ['D']
        self.orig_and_dest = self.origins + self.destinations

        self.act_arcs = set()
        self.err_day_dict = dict()
        self.dict_not_valid_rot = dict()
        self.dict_not_valid_rot_list = dict()

        # Preparing graph structure
        self.build_graph()
        self.first_run = True

    # This function builds the complete dataset of nodes/links
    def build_graph(self):

        # Mapping flights to retimed flights
        self.tasks2rt_tasks = {task: [task + '_' + str(self.delta2pos[delta]) for delta in self.deltas_in_tasks[task]]
                               for task in self.deltas_in_tasks}

        self.rt_tasks2tasks = {rt_task: task for (task, rt_tasks) in self.tasks2rt_tasks.items()
                               for rt_task in rt_tasks}

        self.rt_tasks = [j for i in self.tasks2rt_tasks.values() for j in i if j]

        # Building the links
        # nodes = set([a[ARC_COL['FLIGHT']] + '_' + str(self.delta2pos[a[ARC_COL['DELTA']]])
        #         for a in self.connections] + \
        #          [a[ARC_COL['FLIGHT_CONNECTED']] + '_' + str(self.delta2pos[a[ARC_COL['DELTA_CONNECTED']]])
        #           for a in self.connections])

        ##self.incoming_arcs = {n: list() for n in nodes}
        # self.incoming_arcs = {t: list() for t in self.rt_tasks}
        # self.outgoing_arcs = self.incoming_arcs.copy()
        # self.outgoing_arcs.update({'O': list()})
        # self.incoming_arcs.update({'D': list()})

        self.slacks = dict()
        self.arc_rt_st = dict()
        self.tasks_arcs = set()
        self.incoming_arcs = {'D': list()}
        self.outgoing_arcs = {'O': list()}

        for a in self.connections:
            task_rt = str(a[ARC_COL['FLIGHT']]) + '_' + str(self.delta2pos[a[ARC_COL['DELTA']]])
            task_rt_connected = str(a[ARC_COL['FLIGHT_CONNECTED']]) + '_' + \
                                str(self.delta2pos[a[ARC_COL['DELTA_CONNECTED']]])

            self.tasks_arcs.update({(task_rt, task_rt_connected)})
            self.arc_rt_st.update({(task_rt, task_rt_connected): self.is_retiming_arc(task_rt, task_rt_connected)})
            self.slacks.update({(task_rt, task_rt_connected): a[ARC_COL['SLACK']]})
            self.incoming_arcs.update({task_rt: list()})
            self.incoming_arcs.update({task_rt_connected: list()})
            self.outgoing_arcs.update({task_rt: list()})
            self.outgoing_arcs.update({task_rt_connected: list()})

        # Links connecting the base with initial tasks
        self.arcs_orig_2_tasks = {(o, t) for o in self.get_origins() for t in self.rt_tasks
                                  if self.get_task_dep_airport(t) in self.bases}
        self.arc_rt_st.update({a: self.is_retiming_task(a[1]) for a in self.arcs_orig_2_tasks})

        # Links connecting final tasks with the base
        self.arcs_tasks_2_dest = {(t, d) for d in self.get_destinations() for t in self.rt_tasks
                                  if self.get_task_arr_airport(t) in self.bases}

        self.arc_rt_st.update({a: self.is_retiming_task(a[0]) for a in self.arcs_tasks_2_dest})

        self.tasks2rt_tasks.update({i: i for i in self.orig_and_dest})
        self.rt_tasks2tasks.update({i: i for i in self.orig_and_dest})

        links23 = self.arcs_orig_2_tasks.union(self.arcs_tasks_2_dest)
        self.id_arcs = sorted(self.tasks_arcs.union(links23))
        for a in self.id_arcs:
            self.incoming_arcs[a[1]].append(a[0])
            self.outgoing_arcs[a[0]].append(a[1])
            # print(self.deltas_in_tasks[a[0]])

        self.rt_arcs = [a for a in self.arc_rt_st if self.arc_rt_st[a]]
        self.slacks.update({a: 0 for a in links23})
        self.total_slack = sum(self.slacks[a] for a in self.get_arcs())

        self.connected_rt_flights = set(self.incoming_arcs.keys()).intersection(set(self.outgoing_arcs.keys()))

        self.connected_tasks = list(set([self.rt_tasks2tasks[rt] for rt in self.connected_rt_flights]))

        self.disconnected_tasks = set(self.tasks).difference(set(self.connected_tasks))

    def are_there_tasks(self):
        return len(self.tasks) > 1

    def is_origin(self, id_task):
        return self.tasks2types[self.rt_tasks2tasks[id_task]] == 'SOURCE'

    def is_destination(self, id_task):
        return self.tasks2types[self.rt_tasks2tasks[id_task]] == 'SINK'

    def is_retiming_task(self, task):
        return not self.is_origin(task) and not self.is_destination(task) and self.pos2delta[int(task[-1])] != 0

    def is_retiming_arc(self, task, task_connected):
        return self.is_retiming_task(task) or self.is_retiming_task(task_connected)

    def inc_opt_step(self):
        self.n_opt_steps += 1

    def set_obj_type(self, obj_type):
        self.obj_type = obj_type

    def enable_kpi_limit(self, kpi):
        self.kpi_limit_stat[kpi] = True

    def disable_kpi_limit(self, kpi):
        self.kpi_limit_stat[kpi] = False

    def get_origins(self):
        return self.origins

    def get_destinations(self):
        return self.destinations

    def get_orig_to_tasks_arcs(self):
        return self.arcs_orig_2_tasks

    def get_n_opt_steps(self):
        return self.n_opt_steps

    def get_arcs(self):
        return self.id_arcs

    def get_tasks_arcs(self):
        return self.tasks_arcs

    def get_act_arcs(self):
        return self.act_arcs

    def get_rt_arcs(self):
        return self.rt_arcs

    def get_obj_type(self):
        return self.obj_type

    def get_rt_tasks(self):
        return self.rt_tasks

    def get_task(self, rt_task):
        return self.rt_tasks2tasks[rt_task]

    def get_task_dep_airport(self, rt_task):
        return self.tasks_airports[rt_task][0]

    def get_task_arr_airport(self, rt_task):
        return self.tasks_airports[rt_task][1]

    def get_task_dep_time(self, rt_task):
        return self.tasks_airports[rt_task][2]

    def get_task_arr_time(self, rt_task):
        return self.tasks_airports[rt_task][3]

    def get_task_delta_time(self, rt_task):
        return self.pos2delta[int(rt_task[-1])]

    def get_arc_delta_value(self, id_task):
        pos = id_task[len(self.rt_tasks2tasks[id_task]) + 1:len(id_task)]

        if len(pos) == 0:
            return 0

        return abs(self.pos2delta[int(pos)])

    def save_current_solution(self):
        self.act_arcs_copy = self.act_arcs.copy()
        self.n_used_ac_copy = self.kpi_val['n_used_ac']

    def release_previous_solution(self):
        self.act_arcs = self.act_arcs_copy.copy()
        self.kpi_val['n_used_ac'] = self.n_used_ac_copy

    def print_kpis(self):
        for kpi in self.kpi_val:
            self.print_kpi(kpi)

    def print_kpi(self, kpi):
        print(kpi, ': ', self.kpi_val[kpi])
