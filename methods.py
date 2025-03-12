from cplex import Cplex
from group_data import AircraftRoutingParams
from group_output import AircraftRoutingOutputs
from model import build_base_ilp, update_base_ilp, solve_ilp
from project_constants import OBJ_FUN, ERROR_CODE, OBJ2ERR_CODE, OBJ2ERR_MSG, OBJ_2_KPI, OBJ2ACK_MSG


def min_n_unsch_flights(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs):
    print('Minimizing # of unscheduled flights...')
    mdl = optimize_objective(params=params, mdl=[], outputs=outputs, objective=OBJ_FUN['MIN_N_UNSCH_F'])

    if outputs.no_feasible_solution():
        print('# of unscheduled flights minimization infeasible')
        outputs.code_err.append(ERROR_CODE['N_UNSCH_INFEAS'])
        return None

    return mdl


def min_n_aircrafts(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs, mdl: Cplex):
    print('Minimizing # of used aircrafts...')

    mdl = optimize_objective(params=params, mdl=mdl, outputs=outputs, objective=OBJ_FUN['MIN_N_AC'])

    if outputs.no_feasible_solution():
        print('# of aircrafts minimization infeasible')
        outputs.code_err.append(ERROR_CODE['N_AC_INFEAS'])
        return None

    return mdl


# This function determines the first Optimal Set of Rotations (OSRs) with minimum re-timing
def min_rt_changes(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs, mdl: Cplex):
    print('Minimizing # of re-timings...')

    params.kpi_limit_stat["n_used_ac"] = True
    params.n_used_ac_bck = True

    optimize_objective(params=params, mdl=mdl, outputs=outputs, objective=OBJ_FUN['MIN_N_RT'])

    if outputs.no_feasible_solution():
        print('# of re-timings minimization infeasible')
        outputs.code_err.append(ERROR_CODE['RT_CH_INFEAS'])
        return None


def min_rt_value(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs, mdl: Cplex):
    print('Minimizing total re-timing value...')
    optimize_objective(params=params, mdl=mdl, outputs=outputs, objective=OBJ_FUN['MIN_RT_VALUE'])

    if not params.quality_chk_status:
        return None

    if outputs.no_feasible_solution():
        print('RT value minimization infeasible')
        outputs.code_err.append(ERROR_CODE['RT_VAL_INFEAS'])
        return None


def min_total_slack(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs, mdl: Cplex):
    print('Minimizing the total slack...')
    mdl = optimize_objective(params=params, mdl=mdl, outputs=outputs, objective=OBJ_FUN['MIN_TOTAL_SLACK'])

    if mdl is None:
        return None

    if outputs.no_feasible_solution():
        print('Total slack minimization infeasible')
        outputs.code_err.append(ERROR_CODE['MIN_TOTAL_INFEAS'])
        return None


def max_total_slack(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs, mdl: Cplex):
    print('Maximizing the total slack...')
    mdl = optimize_objective(params=params, mdl=mdl, outputs=outputs, objective=OBJ_FUN['MAX_TOTAL_SLACK'])

    if mdl is None:
        return None

    if outputs.no_feasible_solution():
        print('Total slack maximization infeasible')
        outputs.code_err.append(ERROR_CODE['MAX_TOTAL_INFEAS'])
        return None


def optimize_objective(*, params: AircraftRoutingParams, outputs: AircraftRoutingOutputs, mdl: Cplex,
                       objective: int) -> Cplex:
    params.set_obj_type(objective)
    params.inc_opt_step()

    if params.first_opt_goal:
        mdl = build_base_ilp(params=params, outputs=outputs)
        params.first_opt_goal = False
    else:
        update_base_ilp(params=params, outputs=outputs, mdl=mdl)

    if mdl is None:
        return None

    solve_ilp(mdl=mdl, params=params, outputs=outputs)

    if params.set_kpi_limit:
        print(' Enabling KPI: ', OBJ_2_KPI[objective])
        params.enable_kpi_limit(OBJ_2_KPI[objective])

    return mdl
