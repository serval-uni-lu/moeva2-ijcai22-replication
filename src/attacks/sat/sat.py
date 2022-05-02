import gurobipy as gp
import numpy as np
from gurobipy import GRB, abs_, max_, LinExpr
from joblib import Parallel, delayed
from tqdm import tqdm

from src.attacks.moeva2.constraints import Constraints
from src.attacks.moeva2.feature_encoder import get_encoder_from_constraints

type_mask_transform = {
    "real": GRB.CONTINUOUS,
    "int": GRB.INTEGER,
    "ohe0": GRB.INTEGER,
    "ohe1": GRB.INTEGER,
    "ohe2": GRB.INTEGER,
}

SAFETY_DELTA = 0.0000001


class SatAttack:
    def __init__(
        self,
        constraints: Constraints,
        sat_constraints,
        min_max_scaler,
        eps,
        norm,
        n_sample=1,
        verbose=1,
        n_jobs=1,
    ):
        self.constraints = constraints
        self.sat_constraints = sat_constraints
        self.min_max_scaler = min_max_scaler
        self.eps = eps
        self.norm = norm
        self.n_sample = n_sample
        self.verbose = verbose
        self.encoder = get_encoder_from_constraints(self.constraints)
        self.n_jobs = n_jobs

    @staticmethod
    def create_variable(m, x_init, type_mask, lb, ub):

        return [
            m.addVar(
                lb=lb[i],
                ub=ub[i],
                vtype=type_mask_transform[type_mask[i]],
                name=f"f{i}",
            )
            for i, feature in enumerate(x_init)
        ]

    @staticmethod
    def create_mutable_constraints(m, x_init, variables, mutable_mask):
        indexes = np.argwhere(~mutable_mask).reshape(-1)

        for i in indexes:
            m.addConstr(variables[i] == x_init[i], f"mut{i}")

    def create_l_constraints(self, m, variables, x_init, lb, ub, norm="inf"):
        x_init_scaled = self.min_max_scaler.transform(x_init.reshape(1, -1))[0]

        # Minimise distance
        # scaleds = [
        #     (variables[i] - lb[i]) / (ub[i] - lb[i]) for i, _ in enumerate(variables)
        # ]
        #
        # def create_distance_vars(i, scaled):
        #     distance = m.addVar(vtype=GRB.CONTINUOUS, name=f"distance_{i}")
        #     distance_abs = m.addVar(vtype=GRB.CONTINUOUS, name=f"distance_abs_{i}")
        #     m.addConstr(distance == scaled - x_init_scaled[i])
        #     m.addConstr(distance_abs == abs_(distance))
        #     return distance_abs
        #
        # if norm in ["inf", np.inf]:
        #     distances = [
        #         create_distance_vars(i, scaled) for i, scaled in enumerate(scaleds)
        #     ]
        #     distance_max = m.addVar(vtype=GRB.CONTINUOUS, name="distance_max")
        #     m.addConstr(distance_max == max_(distances))
        #     m.setObjective(distance_max, GRB.MINIMIZE)
        if norm in ["inf", np.inf]:
            for i, e in enumerate(variables):
                if lb[i] != ub[i]:
                    scaled = (variables[i] - lb[i]) / (ub[i] - lb[i])
                    m.addConstr(
                        scaled <= x_init_scaled[i] + self.eps - SAFETY_DELTA,
                        f"scaled_{i}",
                    )
                    m.addConstr(
                        scaled >= x_init_scaled[i] - self.eps + SAFETY_DELTA,
                        f"scaled_{i}",
                    )
                else:
                    scaled = 0

        elif norm in ["2", 2]:

            def create_distance_vars(i, scaled):
                distance_l = m.addVar(vtype=GRB.CONTINUOUS, name=f"distance_{i}")
                m.addConstr(distance_l == scaled - x_init_scaled[i])
                return distance_l

            scaleds = [
                (variables[i] - lb[i]) / (ub[i] - lb[i])
                for i, _ in enumerate(variables)
            ]

            distances = [
                create_distance_vars(i, scaled) for i, scaled in enumerate(scaleds)
            ]

            sum_squared = LinExpr()
            for distance in distances:
                sum_squared.add(distance * distance)

            l2_distance = m.addVar(vtype=GRB.CONTINUOUS, name=f"l2_distance")
            m.addGenConstrPow(l2_distance, sum_squared, 2, name="l2_distance")

        else:
            raise NotImplementedError

    @staticmethod
    def apply_hot_start(variables, x_hot_start=None):
        if x_hot_start is not None:
            for i in range(len(variables)):
                variables[i].start = x_hot_start[i]

    def create_model(self, x_init, x_hot_start=None):
        # Pre fetch
        mutable_mask = self.constraints.get_mutable_mask()
        type_mask = self.constraints.get_feature_type()
        lb, ub = self.constraints.get_feature_min_max(dynamic_input=x_init)

        m = gp.Model("mip1")

        # Create variables
        variables = self.create_variable(m, x_init, type_mask, lb, ub)

        # Mutable constraints
        self.create_mutable_constraints(m, x_init, variables, mutable_mask)

        # Constraints
        self.sat_constraints(m, variables)

        # Distance constraints
        self.create_l_constraints(
            m,
            variables,
            x_init,
            self.min_max_scaler.data_min_,
            self.min_max_scaler.data_max_,
        )

        # Hot start
        self.apply_hot_start(variables, x_hot_start)

        return m

    def _one_generate(self, x_init, x_hot_start=None):
        try:

            m = self.create_model(x_init, x_hot_start)
            m.setParam(GRB.Param.PoolSolutions, self.n_sample)
            m.setParam(GRB.Param.PoolSearchMode, 2)
            m.setParam(GRB.Param.NonConvex, 2)
            m.setParam(GRB.Param.NumericFocus, 3)
            m.setParam(GRB.Param.StartNodeLimit, 1000)
            m.setParam(GRB.Param.Threads, 1)
            m.optimize()
            nSolutions = m.SolCount

            def get_vars(e):
                m.setParam(GRB.Param.SolutionNumber, e)
                return [v.X for v in m.getVars()]

            if nSolutions > 0:
                solutions = np.array([get_vars(e) for e in range(nSolutions)])[
                    :, : x_init.shape[0]
                ]
            else:
                solutions = np.array([x_init] * int(self.n_sample))

            # print(solutions.shape)

        except gp.GurobiError as e:
            print("Error code " + str(e.errno) + ": " + str(e))

        except AttributeError:
            print("Encountered an attribute error")

        return solutions

    def generate(self, x_initial, x_hot_start=None):

        x_hot_start_local = [None for _ in range(x_initial.shape[0])]
        if x_hot_start is not None:
            if x_initial.shape != x_hot_start.shape:
                raise ValueError

            x_hot_start_local = x_hot_start

        iterable = enumerate(x_initial)
        if self.verbose > 0:
            iterable = tqdm(iterable, total=len(x_initial))

        # return np.array(
        #     [self._one_generate(x_init, x_hot_start_local[i]) for i, x_init in iterable]
        # )

        # Sequential Run
        if self.n_jobs == 1:
            return np.array(
                [
                    self._one_generate(x_init, x_hot_start_local[i])
                    for i, x_init in iterable
                ]
            )

        # Parallel run
        else:

            return np.array(
                Parallel(n_jobs=self.n_jobs)(
                    delayed(self._one_generate)(x_init, x_hot_start_local[i])
                    for i, x_init in iterable
                )
            )
