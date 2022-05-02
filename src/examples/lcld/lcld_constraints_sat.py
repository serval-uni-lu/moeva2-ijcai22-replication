import pickle

import gurobipy
from gurobipy import quicksum, GRB


def create_constraints(m, vars):

    # g1

    # proxy1 = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"proxy1")
    # proxy1_1 = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"proxy1_1")
    # proxy2 = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"proxy2")
    # proxy3 = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"proxy3")
    # proxy4 = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"proxy4")
    # proxy5 = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"proxy5")

    # Restart
    rate = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"rate")
    m.addConstr(rate * 1200 == vars[2])

    rate1 = m.addVar(vtype=GRB.CONTINUOUS, name=f"rate1")
    m.addConstr(rate1 == rate + 1)

    term_36 = m.addVar(vtype=GRB.BINARY, name=f"term36")
    m.addConstr((term_36 == 0) >> (vars[1] == 36))
    m.addConstr((term_36 == 1) >> (vars[1] == 60))

    # term_60 = m.addVar(vtype=GRB.BINARY, name=f"term60")
    # m.addConstr((term_60 == 0) >> (vars[1] == 60))

    rate1_p30 = m.addVar(vtype=GRB.CONTINUOUS, name=f"rate1_p30")
    m.addGenConstrPow(rate1, rate1_p30, 36)

    rate1_p60 = m.addVar(vtype=GRB.CONTINUOUS, name=f"rate1_p60")
    m.addGenConstrPow(rate1, rate1_p60, 60)

    rate1_p30_1 = m.addVar(vtype=GRB.CONTINUOUS, name=f"rate1_p30_1")
    m.addConstr(rate1_p30_1 == rate1_p30 - 1)

    rate1_p60_1 = m.addVar(vtype=GRB.CONTINUOUS, name=f"rate1_p60_1")
    m.addConstr(rate1_p60_1 == rate1_p60 - 1)

    rate1_p30_rate = m.addVar(vtype=GRB.CONTINUOUS, name=f"rate1_p30_rate")
    m.addConstr(rate1_p30_rate == rate1_p30 * rate)

    rate1_p60_rate = m.addVar(vtype=GRB.CONTINUOUS, name=f"rate1_p60_rate")
    m.addConstr(rate1_p60_rate == rate1_p60 * rate)

    right_30 = m.addVar(vtype=GRB.CONTINUOUS, name=f"right_30")
    m.addConstr(right_30 == vars[0] * rate1_p30_rate)

    right_60 = m.addVar(vtype=GRB.CONTINUOUS, name=f"right_60")
    m.addConstr(right_60 == vars[0] * rate1_p60_rate)

    left_30 = m.addVar(vtype=GRB.CONTINUOUS, name=f"left_30")
    m.addConstr(left_30 == vars[3] * rate1_p30_1)

    left_60 = m.addVar(vtype=GRB.CONTINUOUS, name=f"left_60")
    m.addConstr(left_60 == vars[3] * rate1_p60_1)

    m.addConstr((term_36 == 0) >> (left_30 == right_30))
    m.addConstr((term_36 == 1) >> (left_60 == right_60))

    # numerator = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"numerator")
    # denominator = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"denominator")
    #
    # # numerator 3
    # m.addConstr(rate * 1200 == vars[2])
    # m.addConstr(rate1 == rate + 1)
    #
    #
    #
    # # denominator 1
    # m.addConstr(proxy1_1 == (1 + proxy1))
    #
    # m.addConstr(proxy2 == vars[3] * 100 * (1 + proxy1))
    #
    # # numerator 1 and 2
    # m.addConstr(proxy3 == 100 * vars[0])
    #
    # # numerator 3 and 4
    # m.addConstr(proxy4 == proxy1 * proxy1_1)
    #
    # # numerator 1 2 3 4
    # m.addConstr(proxy5 == proxy3 * proxy4)
    #
    # # m.addConstr(
    # #     (proxy2 ** vars[1] - 1)
    # #     == (proxy5 ** vars[1])
    # # )
    #
    # m.addConstr(vars[3] * 100 * denominator == numerator)
    #
    # g2
    m.addConstr(vars[10] <= vars[14])

    # g3
    m.addConstr(vars[16] <= vars[11])

    # g4
    # m.addConstr(vars[1] - 36 * vars[1] - 60 == 0)

    # # g5
    m.addConstr(vars[20] * vars[6] == vars[0])

    # g6
    m.addConstr(vars[21] * vars[14] == vars[10])

    # g7
    nb_month_100_7 = m.addVar(vtype=GRB.INTEGER, name=f"nb_month_100_7")
    nb_month_1_7 = m.addVar(vtype=GRB.INTEGER, name=f"nb_month_1_7")
    nb_month_all_7 = m.addVar(vtype=GRB.INTEGER, name=f"nb_month_all_7")
    m.addConstr((nb_month_100_7 * 100) <= vars[7])
    m.addConstr((nb_month_100_7 * 100) >= vars[7] - 100)
    m.addConstr(nb_month_1_7 == vars[7] - nb_month_100_7 * 100)
    m.addConstr(nb_month_all_7 == nb_month_100_7 * 12 + nb_month_1_7)

    nb_month_100_9 = m.addVar(vtype=GRB.INTEGER, name=f"nb_month_100_9")
    nb_month_1_9 = m.addVar(vtype=GRB.INTEGER, name=f"nb_month_1_9")
    nb_month_all_9 = m.addVar(vtype=GRB.INTEGER, name=f"nb_month_all_9")
    m.addConstr((nb_month_100_9 * 100) <= vars[9])
    m.addConstr((nb_month_100_9 * 100) >= vars[9] - 100)
    m.addConstr(nb_month_1_9 == vars[9] - nb_month_100_9 * 100)
    m.addConstr(nb_month_all_9 == nb_month_100_9 * 12 + nb_month_1_9)


    # feat1 = (vars[7] // 100) * 12 + (vars[7] % 100)
    # feat2 = (vars[9] // 100) * 12 + (vars[9] % 100)
    #
    m.addConstr(vars[22] == nb_month_all_7 - nb_month_all_9)
    #
    # g8
    m.addConstr(vars[23] * vars[22] == vars[11])

    # g9
    m.addConstr(vars[24] * vars[22] == vars[16])

    # g10
    # ratio = m.addVar(
    #             lb=-1,
    #             name=f"ratio",
    #         )

    denom_zero = m.addVar(vtype=GRB.BINARY, name=f"denom_zero")
    denom_one = m.addVar(vtype=GRB.BINARY, name=f"denom_one")
    proxy = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"proxy2")

    m.addConstr((denom_zero == 0) >> (vars[11] == 0))
    m.addConstr(denom_one == 1 - denom_zero)
    m.addConstr(proxy == (vars[25] * vars[11] - vars[16]))

    m.addConstr((denom_one * (vars[25] + 1)) + (denom_zero * proxy) == 0)
    # m.addConstr(denom_zero == 0 >> vars[25] == -1)
    # m.addConstr(denom_zero == 1 >> vars[25] * vars[11] == vars[16])
    pass
