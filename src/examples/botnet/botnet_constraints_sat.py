import pickle

import gurobipy
from gurobipy import quicksum, GRB, LinExpr


def sum_equals_sum(index_a, index_b, vars):
    expr1 = LinExpr()
    expr2 = LinExpr()

    for i in index_a:
        expr1.add(vars[i])
    for i in index_b:
        expr2.add(vars[i])

    return expr1 == expr2


def define_individual_constraints(m, vars, cons_idx, feat_idx, upper_idx, lower_idx):
    keys = list(feat_idx.keys())

    for i in range(len(upper_idx)):
        key = keys[upper_idx[i]]
        type_lower = keys[lower_idx[i]]
        type_upper = keys[upper_idx[i]]
        for j in range(len(feat_idx[key])):
            port_idx_lower = feat_idx[type_lower][j]
            port_idx_upper = feat_idx[type_upper][j]
            m.addConstr(vars[port_idx_lower] <= vars[port_idx_upper])
            cons_idx += 1
    return cons_idx


def define_individual_constraints_pkts_bytes(m, vars, cons_idx, feat_idx):
    bytes_out = ["bytes_out_sum_s_idx", "bytes_out_sum_d_idx"]
    pkts_out = ["pkts_out_sum_s_idx", "pkts_out_sum_d_idx"]
    for i in range(len(bytes_out)):
        pkts = feat_idx[pkts_out[i]]
        bytes_ = feat_idx[bytes_out[i]]
        for j in range(len(bytes_out[i]) - 2):
            port_idx_pkts = pkts[j]
            port_idx_bytes = bytes_[j]

            m.addConstr(vars[port_idx_bytes] <= vars[port_idx_pkts] * 1500)
            cons_idx += 1
    return cons_idx


def create_constraints(m, vars):

    with open("./data/botnet/feat_idx.pickle", "rb") as f:
        feat_idx = pickle.load(f)

    sum_idx = [0, 3, 6, 12, 15, 18]
    max_idx = [1, 4, 7, 13, 16, 19]
    min_idx = [2, 5, 8, 14, 17, 20]

    # g1
    m.addConstr(
        sum_equals_sum(
            (
                feat_idx["icmp_sum_s_idx"]
                + feat_idx["udp_sum_s_idx"]
                + feat_idx["tcp_sum_s_idx"]
            ),
            (feat_idx["bytes_in_sum_s_idx"] + feat_idx["bytes_out_sum_s_idx"]),
            vars,
        )
    )

    # g2
    m.addConstr(
        sum_equals_sum(
            (
                feat_idx["icmp_sum_d_idx"]
                + feat_idx["udp_sum_d_idx"]
                + feat_idx["tcp_sum_d_idx"]
            ),
            (feat_idx["bytes_in_sum_d_idx"] + feat_idx["bytes_out_sum_d_idx"]),
            vars,
        )
    )

    cons_idx = 3
    cons_idx = define_individual_constraints_pkts_bytes(m, vars, cons_idx, feat_idx)
    cons_idx = define_individual_constraints(
        m, vars, cons_idx, feat_idx, sum_idx, max_idx
    )
    cons_idx = define_individual_constraints(
        m, vars, cons_idx, feat_idx, sum_idx, min_idx
    )
    cons_idx = define_individual_constraints(
        m, vars, cons_idx, feat_idx, max_idx, min_idx
    )
