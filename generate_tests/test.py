import numpy as np
import random
import math
import json
import os
from pathlib import Path
import pulp
from ortools.constraint_solver import pywrapcp
import networkx as nx
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp


def get_task_sizes_pangu(tasks, N, alpha=1):
    problem = pulp.LpProblem("task_distribution", pulp.LpMaximize)
    x0 = pulp.LpVariable("x0", lowBound=0, cat="Integer")
    x1 = pulp.LpVariable("x1", lowBound=0, cat="Integer")
    x2 = pulp.LpVariable("x2", lowBound=0, cat="Integer")
    x3 = pulp.LpVariable("x3", lowBound=0, cat="Integer")
    x4 = pulp.LpVariable("x4", lowBound=0, cat="Integer")
    x5 = pulp.LpVariable("x5", lowBound=0, cat="Integer")

    problem += (
        1 * 8 * x0 + 3 * 8 * x1 + 4 * 8 * x2 + 8 * 16 * x3 + 12 * 16 * x4 + 24 * 16 * x5
    )
    problem += (
        1 * 8 * x0 + 3 * 8 * x1 + 4 * 8 * x2 + 8 * 16 * x3 + 12 * 16 * x4 + 24 * 16 * x5
        <= alpha * N
    )
    problem += 1 * x0 + 3 * x1 + 4 * x2 + 8 * x3 + 12 * x4 + 24 * x5 == tasks

    problem.solve()
    print(
        int(pulp.value(x0)),
        int(pulp.value(x1)),
        int(pulp.value(x2)),
        int(pulp.value(x3)),
        int(pulp.value(x4)),
        int(pulp.value(x5)),
    )
    print(
        1 * int(pulp.value(x0)),
        3 * int(pulp.value(x1)),
        4 * int(pulp.value(x2)),
        8 * int(pulp.value(x3)),
        12 * int(pulp.value(x4)),
        24 * int(pulp.value(x5)),
    )
    res = []
    res += [8] * (int(pulp.value(x0)) * 1)
    res += [8] * (int(pulp.value(x1)) * 3)
    res += [8] * (int(pulp.value(x2)) * 4)
    res += [16] * (int(pulp.value(x3)) * 8)
    res += [16] * (int(pulp.value(x4)) * 12)
    res += [16] * (int(pulp.value(x5)) * 24)
    return res


def get_task_sizes(tasks, N):
    logN = int(np.log2(N)) + 1
    problem = pulp.LpProblem("task_distribution", pulp.LpMaximize)

    variables = pulp.LpVariable.dict("Var", range(logN - 2), lowBound=0, cat="Integer")

    # problem += pulp.lpSum(
    #     [
    #         (2**i if i >= math.floor(logN / 2) + 1 else 2 ** (i - logN)) * variables[i]
    #         for i in range(2, logN)
    #     ]
    # )
    problem += pulp.lpSum([(2 ** (i + 2)) * variables[i] for i in range(0, logN - 2)])
    problem += (
        pulp.lpSum([(2 ** (i + 2)) * variables[i] for i in range(0, logN - 2)]) <= N
    )
    problem += pulp.lpSum([variables[i] for i in range(0, logN - 2)]) == tasks

    problem.solve()
    res = []
    for i, v in enumerate(problem.variables()):
        if int(v.varValue) == 0:
            continue
        res += [2 ** (i + 2)] * int(v.varValue)
    return res


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables: list[cp_model.IntVar]):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.res = []

    def on_solution_callback(self) -> None:
        # self.__solution_count += 1
        # self.res.append([self.value(v) for v in self.__variables])
        # print()
        self.__solution_count += 1
        for v in self.__variables:
            print(f"{v}={self.value(v)}", end=" ")
        print()

    @property
    def solution_count(self) -> int:
        return self.__solution_count


def f(tasks, N):
    logN = int(np.log2(N)) + 1
    model = cp_model.CpModel()
    variables = [model.NewIntVar(0, N, str(i)) for i in range(logN - 2)]
    model.maximize(sum([(2 ** (i + 2)) * variables[i] for i in range(0, logN - 2)]))
    model.add(sum([(2 ** (i + 2)) * variables[i] for i in range(0, logN - 2)]) <= N)
    model.add(sum([variables[i] for i in range(logN - 2)]) == tasks)

    solver = cp_model.CpSolver()

    status = solver.solve(model)

    res = []
    for i in range(logN - 2):
        if solver.value(variables[i]) == 0:
            continue
        res += [2 ** (i + 2)] * solver.value(variables[i])

    M = sum(res)
    start = []
    while M > 0:
        num = 2 ** (int(np.log2(M)))
        start.append(num)
        M -= num
    start = sorted(start)
    start = tuple(start)
    res_set = {start}
    for _ in range(tasks - len(start)):
        new_set = set()
        for distr in res_set:
            for i, x in enumerate(distr):
                if x <= 4:
                    continue
                new_distr = list(distr[:i]) + [x // 2, x // 2] + list(distr[i + 1 :])
                new_distr = tuple(sorted(new_distr))
                new_set.add(new_distr)
        res_set = new_set
    res_set = [list(x) for x in res_set]
    return res_set


res_tests = [
    # [2, 8, 4, [16, 16]],
    # [2, 8, 4, [8, 8]],
    # [2, 8, 4, [4, 4]],
    # [3, 8, 4, [8, 8, 16]],
    # [3, 8, 4, [4, 4, 8]],
    # [3, 8, 4, [4, 4, 4]],
    # [4, 8, 4, [8, 8, 8, 8]],
    # [4, 8, 4, [4, 4, 4, 4]],
    # [4, 8, 4, [4, 4, 4, 4, 4]],
    # [5, 8, 4, [4, 4, 8, 8, 8]],
    # [5, 8, 4, [4, 4, 4, 4, 4]],
    # [5, 8, 4, [4, 4, 4, 4, 4, 4, 4]],
    [2, 16, 8, [64, 64]],
    [2, 16, 8, [32, 32]],
    [2, 16, 8, [8, 32]],
    [3, 16, 8, [32, 32, 64]],
    [3, 16, 8, [16, 16, 32]],
    [3, 16, 8, [8, 16, 16]],
    [4, 16, 8, [16, 16, 32, 64]],
    [4, 16, 8, [16, 16, 16, 16]],
    [4, 16, 8, [8, 8, 8, 16]],
    [5, 16, 8, [16, 16, 16, 16, 64]],
    [5, 16, 8, [8, 8, 8, 8, 32]],
    [5, 16, 8, [8, 8, 8, 8, 8]],
    [6, 16, 8, [8, 8, 8, 8, 32, 64]],
    [6, 16, 8, [8, 8, 8, 8, 16, 16]],
    [6, 16, 8, [4, 4, 8, 8, 8, 8]],
    [7, 16, 8, [16, 16, 16, 16, 16, 16, 32]],
    [7, 16, 8, [8, 8, 8, 8, 8, 8, 16]],
    [7, 16, 8, [4, 4, 4, 4, 8, 8, 8]],
    [8, 16, 8, [16, 16, 16, 16, 16, 16, 16, 16]],
    # [8, 16, 8, [8, 8, 8, 8, 8, 8, 8, 8]],
    # [8, 16, 8, [4, 4, 4, 4, 4, 4, 8, 8]],
    [9, 16, 8, [8, 8, 16, 16, 16, 16, 16, 16, 16]],
    # [9, 16, 8, [4, 4, 8, 8, 8, 8, 8, 8, 8]],
    # [9, 16, 8, [4, 4, 4, 4, 4, 4, 4, 4, 8]],
    # [10, 16, 8, [4, 4, 8, 16, 16, 16, 16, 16, 16, 16]],
    # [10, 16, 8, [4, 4, 4, 4, 8, 8, 8, 8, 8, 8]],
    # [10, 16, 8, [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]],
    # [11, 16, 8, [4, 4, 4, 4, 16, 16, 16, 16, 16, 16, 16]],
    # [11, 16, 8, [4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8]],
    # [11, 16, 8, [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]],
    # [12, 16, 8, [4, 4, 4, 4, 8, 8, 16, 16, 16, 16, 16, 16]],
    # [12, 16, 8, [4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8]],
    # [12, 16, 8, [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]],
    [2, 32, 16, [256, 256]],
    [2, 32, 16, [128, 128]],
    [2, 32, 16, [32, 128]],
    [3, 32, 16, [128, 128, 256]],
    [3, 32, 16, [64, 64, 128]],
    [3, 32, 16, [8, 32, 128]],
    [4, 32, 16, [64, 64, 128, 256]],
    [4, 32, 16, [32, 32, 64, 128]],
    [4, 32, 16, [4, 4, 32, 128]],
    [5, 32, 16, [64, 64, 64, 64, 256]],
    [5, 32, 16, [32, 32, 32, 32, 128]],
    [5, 32, 16, [8, 16, 16, 64, 64]],
    [6, 32, 16, [32, 32, 64, 64, 64, 256]],
    [6, 32, 16, [16, 16, 32, 32, 32, 128]],
    [6, 32, 16, [8, 32, 32, 32, 32, 32]],
    [7, 32, 16, [32, 32, 32, 32, 64, 64, 256]],
    [7, 32, 16, [16, 16, 16, 16, 32, 32, 128]],
    [7, 32, 16, [8, 8, 8, 8, 8, 64, 64]],
    [8, 32, 16, [32, 32, 32, 32, 32, 32, 64, 256]],
    [8, 32, 16, [4, 4, 4, 4, 16, 32, 64, 128]],
    [8, 32, 16, [8, 16, 16, 16, 16, 32, 32, 32]],
    [9, 32, 16, [8, 8, 8, 8, 32, 64, 64, 64, 256]],
    [9, 32, 16, [16, 16, 16, 16, 16, 16, 16, 16, 128]],
    [9, 32, 16, [4, 4, 16, 16, 16, 16, 32, 32, 32]],
    [10, 32, 16, [16, 16, 32, 32, 32, 32, 32, 32, 32, 256]],
    [10, 32, 16, [8, 8, 16, 16, 16, 16, 16, 16, 16, 128]],
    [10, 32, 16, [4, 4, 4, 4, 4, 4, 4, 4, 8, 128]],
    [11, 32, 16, [4, 4, 4, 4, 4, 4, 8, 32, 64, 128, 256]],
    [11, 32, 16, [8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 128]],
    [11, 32, 16, [4, 4, 8, 8, 8, 8, 16, 16, 32, 32, 32]],
    [12, 32, 16, [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 128]],
    [12, 32, 16, [8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 128]],
    [12, 32, 16, [4, 4, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]],
    [13, 32, 16, [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 128]],
    [13, 32, 16, [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32]],
    [13, 32, 16, [4, 4, 8, 8, 16, 16, 16, 16, 16, 16, 16, 16, 16]],
    [14, 32, 16, [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64]],
    [14, 32, 16, [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32]],
    [14, 32, 16, [4, 4, 4, 4, 8, 16, 16, 16, 16, 16, 16, 16, 16, 16]],
    [15, 32, 16, [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64]],
    [15, 32, 16, [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32]],
    [15, 32, 16, [4, 4, 4, 4, 4, 4, 16, 16, 16, 16, 16, 16, 16, 16, 16]],
    #
    [2, 8, 4, [8, 16]],
    [3, 8, 4, [4, 4, 16]],
    [4, 8, 4, [4, 4, 8, 8]],
    [5, 8, 4, [4, 4, 4, 4, 8]],
    [2, 16, 8, [32, 64]],
    [3, 16, 8, [16, 16, 64]],
    [4, 16, 8, [8, 8, 16, 64]],
    [5, 16, 8, [8, 8, 8, 8, 64]],
    [6, 16, 8, [4, 4, 4, 4, 16, 64]],
    [7, 16, 8, [4, 4, 8, 8, 8, 32, 32]],
    [8, 16, 8, [8, 8, 8, 8, 8, 8, 16, 32]],
    [2, 32, 16, [128, 256]],
    [3, 32, 16, [64, 64, 256]],
    [4, 32, 16, [32, 32, 64, 256]],
    [5, 32, 16, [32, 32, 32, 32, 256]],
    [6, 32, 16, [16, 16, 32, 32, 32, 256]],
    [7, 32, 16, [16, 16, 16, 16, 32, 32, 256]],
    [8, 32, 16, [4, 4, 4, 4, 16, 32, 64, 256]],
    #
    [4, 16, 8, [32, 32, 32, 32]],
    [5, 16, 8, [8, 8, 16, 32, 64]],
    [5, 16, 8, [16, 16, 32, 32, 32]],
    [6, 16, 8, [4, 4, 8, 16, 32, 64]],
    [6, 16, 8, [8, 8, 16, 16, 16, 64]],
    [6, 16, 8, [8, 8, 16, 32, 32, 32]],
    [6, 16, 8, [16, 16, 16, 16, 32, 32]],
    #
]

new_tests = [
    [3, 16, 8, [16, 32, 64], 0],
    [3, 16, 8, [16, 16, 64], 1],
    [3, 16, 8, [32, 32, 32], 2],
    [3, 16, 8, [8, 8, 64], 3],
    [3, 16, 8, [16, 32, 32], 4],
    [4, 16, 8, [16, 32, 32, 32], 0],
    [4, 16, 8, [8, 8, 32, 64], 1],
    [4, 16, 8, [16, 16, 16, 64], 2],
    [4, 16, 8, [16, 16, 32, 32], 3],
    [4, 16, 8, [8, 8, 16, 64], 4],
    [4, 16, 8, [16, 16, 16, 32], 5],
    [4, 16, 8, [4, 4, 8, 64], 6],
    [4, 16, 8, [8, 8, 32, 32], 7],
    [5, 16, 8, [4, 4, 8, 32, 64], 0],
    [5, 16, 8, [16, 16, 16, 32, 32], 1],
    [5, 16, 8, [8, 8, 32, 32, 32], 2],
    [5, 16, 8, [8, 8, 16, 16, 64], 3],
    [5, 16, 8, [8, 8, 16, 32, 32], 4],
    [5, 16, 8, [4, 4, 8, 16, 64], 5],
    [5, 16, 8, [16, 16, 16, 16, 32], 6],
    [5, 16, 8, [8, 8, 8, 8, 64], 7],
    [5, 16, 8, [4, 4, 4, 4, 64], 8],
    [5, 16, 8, [8, 8, 16, 16, 32], 9],
    [5, 16, 8, [16, 16, 16, 16, 16], 10],
    [5, 16, 8, [4, 4, 8, 32, 32], 11],
    [6, 16, 8, [4, 4, 8, 16, 16, 64], 0],
    [6, 16, 8, [4, 4, 8, 32, 32, 32], 1],
    [6, 16, 8, [16, 16, 16, 16, 16, 32], 2],
    [6, 16, 8, [4, 4, 4, 4, 32, 64], 3],
    [6, 16, 8, [8, 8, 16, 16, 32, 32], 4],
    [6, 16, 8, [8, 8, 8, 8, 16, 64], 5],
    [6, 16, 8, [8, 8, 8, 8, 32, 32], 6],
    [6, 16, 8, [4, 4, 8, 8, 8, 64], 7],
    [6, 16, 8, [4, 4, 8, 16, 32, 32], 8],
    [6, 16, 8, [16, 16, 16, 16, 16, 16], 9],
    [6, 16, 8, [4, 4, 4, 4, 16, 64], 10],
    [6, 16, 8, [8, 8, 16, 16, 16, 32], 11],
    [6, 16, 8, [4, 4, 8, 16, 16, 32], 12],
    [6, 16, 8, [8, 8, 8, 8, 16, 32], 13],
    [6, 16, 8, [4, 4, 4, 4, 32, 32], 14],
    [6, 16, 8, [8, 8, 16, 16, 16, 16], 15],
    [7, 16, 8, [8, 8, 8, 8, 16, 32, 32], 0],
    [7, 16, 8, [4, 4, 8, 16, 16, 32, 32], 1],
    [7, 16, 8, [4, 4, 8, 8, 8, 16, 64], 2],
    [7, 16, 8, [8, 8, 8, 8, 8, 8, 64], 3],
    [7, 16, 8, [16, 16, 16, 16, 16, 16, 16], 4],
    [7, 16, 8, [4, 4, 4, 4, 16, 16, 64], 5],
    [7, 16, 8, [4, 4, 4, 4, 32, 32, 32], 6],
    [7, 16, 8, [8, 8, 16, 16, 16, 16, 32], 7],
    [7, 16, 8, [8, 8, 8, 8, 16, 16, 32], 8],
    [7, 16, 8, [8, 8, 16, 16, 16, 16, 16], 9],
    [7, 16, 8, [4, 4, 4, 4, 16, 32, 32], 10],
    [7, 16, 8, [4, 4, 4, 4, 8, 8, 64], 11],
    [7, 16, 8, [4, 4, 8, 16, 16, 16, 32], 12],
    [7, 16, 8, [4, 4, 8, 8, 8, 32, 32], 13],
    [7, 16, 8, [8, 8, 8, 8, 16, 16, 16], 14],
    [7, 16, 8, [4, 4, 8, 16, 16, 16, 16], 15],
    [7, 16, 8, [4, 4, 8, 8, 8, 16, 32], 16],
    [7, 16, 8, [8, 8, 8, 8, 8, 8, 32], 17],
    [7, 16, 8, [4, 4, 4, 4, 16, 16, 32], 18],
    [8, 16, 8, [4, 4, 8, 8, 8, 8, 8, 64], 0],
    [8, 16, 8, [8, 8, 8, 8, 8, 8, 32, 32], 1],
    [8, 16, 8, [4, 4, 4, 4, 8, 8, 16, 64], 2],
    [8, 16, 8, [4, 4, 8, 16, 16, 16, 16, 32], 3],
    [8, 16, 8, [8, 8, 8, 8, 16, 16, 16, 32], 4],
    [8, 16, 8, [4, 4, 8, 8, 8, 16, 32, 32], 5],
    [8, 16, 8, [8, 8, 16, 16, 16, 16, 16, 16], 6],
    [8, 16, 8, [4, 4, 4, 4, 16, 16, 32, 32], 7],
    [8, 16, 8, [4, 4, 4, 4, 16, 16, 16, 32], 8],
    [8, 16, 8, [8, 8, 8, 8, 8, 8, 16, 32], 9],
    [8, 16, 8, [4, 4, 4, 4, 8, 8, 32, 32], 10],
    [8, 16, 8, [4, 4, 4, 4, 4, 4, 8, 64], 11],
    [8, 16, 8, [4, 4, 8, 16, 16, 16, 16, 16], 12],
    [8, 16, 8, [8, 8, 8, 8, 16, 16, 16, 16], 13],
    [8, 16, 8, [4, 4, 8, 8, 8, 16, 16, 32], 14],
    [8, 16, 8, [4, 4, 4, 4, 16, 16, 16, 16], 15],
    [8, 16, 8, [4, 4, 8, 8, 8, 8, 8, 32], 16],
    [8, 16, 8, [8, 8, 8, 8, 8, 8, 16, 16], 17],
    [8, 16, 8, [4, 4, 4, 4, 8, 8, 16, 32], 18],
    [8, 16, 8, [4, 4, 8, 8, 8, 16, 16, 16], 19],
]

tls = [[3, 16, 8], [4, 16, 8], [5, 16, 8], [6, 16, 8], [7, 16, 8], [8, 16, 8]]
res = []
counter = {}
for i, (tasks, leaves, spines) in enumerate(tls):
    res_set = set()
    for N in [leaves * (spines - 1), leaves * (spines - 2), leaves * (spines - 3)]:
        task_distributions = f(tasks, N)
        for task_distr in task_distributions:
            task_distr_tuple = tuple(task_distr)
            if (tasks, leaves, spines, task_distr_tuple) in res_set:
                continue
            res_set.add((tasks, leaves, spines, task_distr_tuple))
            if ((tasks, leaves, spines)) not in counter:
                counter[(tasks, leaves, spines)] = -1
            counter[(tasks, leaves, spines)] += 1
            res.append(
                [tasks, leaves, spines, task_distr, counter[(tasks, leaves, spines)]]
            )
for x in res:
    print(f"{x}, ")


# def ranking_is_colored(t, l, s, ranking):
#     G = nx.Graph()
#     leaves_host_number_on_tasks = [
#         [(task, ranking[task][leaf]) for task in range(t)] for leaf in range(l)
#     ]
#     for leaf in leaves_host_number_on_tasks:
#         for task_id, size in leaf:
#             for i in range(1, size + 1):
#                 for j in range(i + 1, size + 1):
#                     G.add_edge(f"{task_id}_{i}", f"{task_id}_{j}")
#         for from_ in range(0, len(leaf)):
#             for to_ in range(from_ + 1, len(leaf)):
#                 task_from, size_from = leaf[from_]
#                 task_to, size_to = leaf[to_]
#                 for i in range(1, size_from + 1):
#                     for j in range(1, size_to + 1):
#                         G.add_edge(f"{task_from}_{i}", f"{task_to}_{j}")

#     model = cp_model.CpModel()

#     # search_parameters = pywrapcp.DefaultRoutingSearchParameters()
#     # search_parameters.time_limit.seconds = 300

#     node_order = {node: i for i, node in enumerate(G.nodes())}
#     variables = [model.NewIntVar(1, s, node) for node in G.nodes()]
#     for u, v in G.edges():
#         model.add(variables[node_order[u]] != variables[node_order[v]])
#     solver = cp_model.CpSolver()
#     status = solver.solve(model)
#     return (status == cp_model.OPTIMAL) or (status == cp_model.FEASIBLE)


# print(
#     ranking_is_colored(
#         5,
#         8,
#         4,
#         [
#             [0, 0, 0, 0, 0, 1, 0, 3],
#             [0, 3, 0, 0, 0, 0, 0, 1],
#             [0, 1, 4, 0, 0, 3, 0, 0],
#             [0, 0, 0, 4, 0, 0, 4, 0],
#             [4, 0, 0, 0, 4, 0, 0, 0],
#         ]
#     )
# )
