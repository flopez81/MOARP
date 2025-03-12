from cplex import Cplex
import datetime as dt
import multiprocessing
from group_data import AircraftRoutingParams
from group_output import AircraftRoutingOutputs
from cplex.exceptions import CplexError
from numpy import array_equal
from project_constants import CPX_NCPUs, CPX_TIME_LIMIT, CPX_PARALLEL_MODE, OBJ_FUN, ERROR_CODE, \
    SAVE_STATUS, DOM_SIGNS, DOM_OBJS, BASE_CAP


# This function builds the Integer Linear Program representing the Daily Network Generator using CPLEX library
def build_base_ilp(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs) -> Cplex:
    print(" Building ilp from scratch...")

    try:
        # Creating model object
        mdl = Cplex()

        # Setting the proper objective function
        set_objective(params=params, mdl=mdl)
        mdl.objective.set_sense(mdl.objective.sense.minimize)

        params.lin_expr = list()
        params.senses = []
        params.rhs = []

        # Setting the base constraints if needed
        set_base_constraints(params=params, outputs=outputs)

        mdl.linear_constraints.add(lin_expr=params.lin_expr, senses=params.senses, rhs=params.rhs)

        return mdl

    except CplexError as exc:
        raise ValueError(str(exc))
        return None


# This function establishes the proper objective function
def set_objective(*, params: AircraftRoutingParams, mdl: Cplex) -> None:
    params.n_arcs = len(params.get_arcs())

    # Defining the x-variables
    lb = [0] * params.n_arcs
    ub = [1] * params.n_arcs
    types = ["B"] * params.n_arcs
    params.x = {a: i for i, a in enumerate(params.get_arcs())}

    n_arcs = len(params.x)
    n_tasks = len(params.connected_tasks)
    lb += [0]*n_tasks
    ub += [1]*n_tasks
    types += ["C"]*n_tasks
    params.y = {t: i + n_arcs for i, t in enumerate(params.connected_tasks)}
    obj = [0.0]*n_arcs + [1.0]*n_tasks

    # Passing all the data to CPLEX object
    mdl.variables.add(obj=obj, lb=lb, ub=ub, types=types)


# This function establishes the base constraints of the model
def set_base_constraints(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs):

    # sum_{a in delta-(v)} x_a = sum_{a in delta+(v)} x_a, forall v in V
    for v in params.connected_rt_flights:
        vars = [params.x[(v1, v)] for v1 in params.incoming_arcs[v]] + \
               [params.x[(v, v1)] for v1 in params.outgoing_arcs[v]]
        coef = [1.0] * len(params.incoming_arcs[v]) + [-1.0] * len(params.outgoing_arcs[v])
        params.lin_expr.append([vars, coef])

    n_connected_rt_flights = len(params.connected_rt_flights)
    params.senses.extend(["E"] * n_connected_rt_flights)
    params.rhs.extend([0] * n_connected_rt_flights)

    # sum_{j in \delta_{V[t]}} x_jt + y_t = 1, forall t in T
    for t in params.connected_tasks:
        vars = [params.x[(v, v1)] for v in params.tasks2rt_tasks[t] for v1 in params.outgoing_arcs[v]]
        coef = [1.0] * len(vars)
        vars += [params.y[t]]
        coef += [1.0]
        params.lin_expr.append([vars, coef])

    n_connected_tasks = len(params.connected_tasks)
    params.senses.extend(["E"] * n_connected_tasks)
    params.rhs.extend([1] * n_connected_tasks)

    # sum_{a in A: i(a)=o, da(j(a))=b} x_a = sum_{a in A: aa(i(a))=b, j(a)=d} x_a, forall b in B
    for b in params.bases:
        vars1 = [params.x[a] for a in params.arcs_orig_2_tasks if params.get_task_dep_airport(a[1]) == b]
        vars2 = [params.x[a] for a in params.arcs_tasks_2_dest if params.get_task_arr_airport(a[0]) == b]
        coef1 = [1.0]*len(vars1)
        coef2 = [-1.0]*len(vars2)
        params.lin_expr.append([vars1+vars2, coef1+coef2])

    n_bases = len(params.bases)
    params.senses.extend(["E"] * n_bases)
    params.rhs.extend([0] * n_bases)

    # sum_{a in A: i(a)=o, da(j(a))=b} x_a <= q_b, forall b in B
    for b in params.bases:
        vars = [params.x[a] for a in params.arcs_orig_2_tasks if params.get_task_dep_airport(a[1]) == b]
        coef = [1.0]*len(vars)
        params.lin_expr.append([vars, coef])
        params.rhs.append(BASE_CAP[params.fleet][b])

    params.senses.extend(["L"] * n_bases)

    #  Imposing the maximum number of aircrafts that can be used
    if params.kpi_limit_stat["n_used_ac"]:
        set_n_ac_limitation(params=params, outputs=outputs)

    # Imposing the maximum number of re-timings
    if params.kpi_limit_stat["N_RT"]:
        set_n_rt_limitation(params=params, outputs=outputs)

    return 0


def update_base_ilp(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs, mdl: Cplex) -> None:
    print(" Updating the ilp...")
    try:
        update_objective(params=params, mdl=mdl)

        params.lin_expr = list()
        params.senses = []
        params.rhs = []

        # Imposing the total number of flights that can be scheduled
        if params.kpi_limit_stat["n_unsch_f"]:
            set_n_unsch_f_limitation(params=params, outputs=outputs)

        #  Imposing the total number of aircrafts that can be used according to the previously minimum found
        if params.kpi_limit_stat["n_used_ac"]:
            set_n_ac_limitation(params=params, outputs=outputs)

        # Imposing the total number of re-timings according to the previously minimum found
        if params.kpi_limit_stat["N_RT"]:
            set_n_rt_limitation(params=params, outputs=outputs)

        mdl.linear_constraints.add(lin_expr=params.lin_expr, senses=params.senses, rhs=params.rhs)

    except e:
        raise e


def update_objective(*, params: AircraftRoutingParams, mdl: Cplex) -> None:

    if params.get_obj_type() == OBJ_FUN['MIN_N_AC']:
        tuples = [(params.x[a], 1.0) for a in params.arcs_orig_2_tasks]
        tuples += [(params.y[t], 0.0) for t in params.connected_tasks]
    elif params.get_obj_type() == OBJ_FUN['MIN_N_RT']:
        tuples = [(params.x[a], 1.0) if params.arc_rt_st[a] else (params.x[a], 0.0) for a in params.get_arcs()]
    elif params.get_obj_type() == OBJ_FUN['MIN_TOTAL_SLACK']:
        tuples = [(params.x[a], params.slacks[a]) for a in params.get_arcs()]
    elif params.get_obj_type() == OBJ_FUN['MAX_TOTAL_SLACK']:
        tuples = [(params.x[a], - params.slacks[a]) for a in params.get_arcs()]
    else:
        print('Objective {} is not implemented'.format(params.get_obj_type()))
        raise

    mdl.objective.set_linear(tuples)


def set_n_unsch_f_limitation(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs) -> None:
    print(" Setting the n_unsch_f limitation")
    vars = [params.y[t] for t in params.connected_tasks]
    coef = [1.0] * len(vars)

    params.lin_expr.append([vars, coef])
    params.senses.append("L")
    params.rhs.append(float(outputs.kpi['N_UNSCH_F']))

    params.kpi_limit_stat["n_unsch_f"] = False


def set_n_ac_limitation(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs) -> None:
    print(" Setting the n_ac limitation")

    vars = [params.x[a] for a in params.get_orig_to_tasks_arcs()]
    coef = [1.0] * len(vars)

    params.lin_expr.append([vars, coef])
    params.senses.append("L")
    params.rhs.append(float(outputs.kpi['N_AC']))

    params.kpi_limit_stat["n_used_ac"] = False


def set_n_rt_limitation(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs) -> None:

    print(" Setting the n_rt limitation")
    vars = [params.x[a] for a in params.get_rt_arcs()]
    coef = [1.0] * len(vars)

    params.lin_expr.append([vars, coef])
    params.senses.append("L")
    params.rhs.append(outputs.kpi["N_RT"])

    params.kpi_limit_stat["N_RT"] = False


def set_rt_val_limitation(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs) -> None:

    print(" Setting the rt_val limitation")
    vars = [params.x[a] for a in params.get_rt_arcs()]
    coef = [params.get_arc_delta_value(a[1]) for a in params.get_rt_arcs()]

    params.lin_expr.append([vars, coef])
    params.senses.append("L")
    params.rhs.append(outputs.kpi["RT_VAL"])

    params.kpi_limit_stat["RT_VAL"] = False


def set_min_total_slack_limitation(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs) -> None:
    print(' Setting the minimum total slack limitation...')

    vars = [params.x[a] for a in params.get_arcs()]
    coef = [params.slacks[a] for a in params.get_arcs()]

    params.lin_expr.append([vars, coef])
    params.senses.append("G")
    params.rhs.append(outputs.kpi["MIN_TOTAL_SLACK"])

    params.kpi_limit_stat["TOTAL_SLACK"] = False


def set_max_total_slack_limitation(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs) -> None:
    print(' Setting the maximum total slack limitation...')

    vars = [params.x[a] for a in params.get_arcs()]
    coef = [params.slacks[a] for a in params.get_arcs()]

    params.lin_expr.append([vars, coef])
    params.senses.append("L")
    params.rhs.append(outputs.kpi["MAX_TOTAL_SLACK"])

    params.kpi_limit_stat["TOTAL_SLACK"] = False


# This function solves the ILP previously built using CPLEX library and returns the links contained in the rotations
def solve_ilp(*, mdl: Cplex, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs) -> Cplex:

    print(" Solving the ilp...")

    try:
        # Tuning CPLEX parameters
        set_cplex_parameters(params=params, mdl=mdl)

        params.dict_not_valid_rot = params.dict_not_valid_rot_list.copy()
        params.dict_not_valid_rot_list = dict()
        params.act_arcs = list()

        runtime = dt.timedelta(minutes=0)
        outputs.update_ilp_output(obj_type=params.get_obj_type(), n_opt_steps=params.get_n_opt_steps())

        prev1 = dt.datetime.now()
        mdl.solve()
        runtime += dt.datetime.now() - prev1

        if mdl.solution.status[mdl.solution.get_status()] == "MIP_infeasible":
            print(' Infeasible solution...')
            return None

        print(' ilp solved. Postprocessing solution...')
        n_vars = mdl.variables.get_num()
        outputs.var_2_val = {c: mdl.solution.get_values(c) for c in range(n_vars)}
        outputs.act_arcs = [a for a in params.get_arcs() if outputs.var_2_val[params.x[a]] > 0.9]

        outputs.act_y = [outputs.var_2_val[params.y[t]] for t in params.connected_tasks
                         if outputs.var_2_val[params.y[t]] > 0.1]

        outputs.calc_kpis1(params=params)
        outputs.update_per_dict(mdl=mdl, cpu_time=round(runtime.total_seconds(), 0))
        if outputs.output_save_status:
            not_repeated = True
            if outputs.output_save_status == SAVE_STATUS['BACKUP'] and outputs.dict_KPIs_temp['DATE']:
                for s in range(len(outputs.dict_KPIs_temp['DATE'])):
                    if array_equal([outputs.dict_KPIs_temp[obj][s] for obj in DOM_OBJS],
                                   [outputs.kpi[obj]*DOM_SIGNS[obj] for obj in DOM_OBJS]):
                        print(' Solution Repeated')
                        not_repeated = False
                        break

            if not_repeated:
                params.n_opt_steps += 1
                outputs.n_opt_steps = params.n_opt_steps
                build_rotations(params=params, outputs=outputs)
                outputs.update_kpis_dict()

        return mdl

    except CplexError as ex:
        print(ex)
        return None


# This routine defines the tuning parameters of CPLEX solver
def set_cplex_parameters(params: AircraftRoutingParams, mdl: Cplex):

    # Indicate CPLEX name and route
    cplex_file = open('./logs/CPLEX_' + str(params.day)[:10] + '_' + params.fleet + '.log', "w")
    mdl.set_results_stream(cplex_file)

    # Establishing the starting search strategy
    mdl.parameters.advance.set(0)

    # Limiting computing time
    mdl.parameters.timelimit.set(CPX_TIME_LIMIT)

    # Displaying detailed information in case of infeasibility
    mdl.parameters.conflict.display.set(2)

    # Using full solve to widen the search for infeasibilities
    # mdl.parameters.conflict.algorithm.set(6)

    mdl.parameters.mip.display.set(0)  # Disabling log output 5 will enable it

    # mdl.parameters.cpumask.set(CPX_NCPUs)
    mdl.parameters.parallel.set(CPX_PARALLEL_MODE)
    mdl.parameters.threads.set(multiprocessing.cpu_count())  # Enabling all available threads


# Postprocessing the cplex solution
def build_rotations(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs):

    xa = [a for a in outputs.act_arcs]
    xo = [a[0] for a in outputs.act_arcs if params.is_origin(a[0])]
    outputs.ini_current_rots()

    rot_id = 0
    total_extra_tat = 0
    params.n_flights = 0
    while xo:
        f = xo.pop(0)
        outputs.order = 1
        rot_arcs = list()
        rot_tasks = list()

        entered = True

        rot_id += 1
        rot_extra_tat = 0

        while entered:
            entered = False

            for a in xa:
                if a[0] == f:
                    f = a[1]
                    xa.remove(a)
                    rot_arcs.append(a)

                    if not params.is_destination(a[1]):
                        entered = True
                        params.n_flights += 1
                        rot_tasks.append(a[1])
                        rot_extra_tat += params.slacks[a]

                        t = params.rt_tasks2tasks[a[1]]

                        delta = params.pos2delta[int(a[1][len(params.rt_tasks2tasks[a[1]]) + 1:len(a[1])])]
                        outputs.update_rots_dict(id_task=t,
                                                 dep_airport=params.get_task_dep_airport(a[1]),
                                                 arr_airport=params.get_task_arr_airport(a[1]),
                                                 std=params.get_task_dep_time(a[1]),
                                                 sta=params.get_task_arr_time(a[1]),
                                                 rot=rot_id,
                                                 arc_slack=params.slacks[a],
                                                 delta=delta)
                    break

        total_extra_tat += rot_extra_tat
