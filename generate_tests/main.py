import numpy as np
import random
import math
import json
import os
from pathlib import Path
import pulp
from ortools.constraint_solver import pywrapcp
from ortools.sat.python import cp_model
import networkx as nx
import concurrent.futures
import threading
from multiprocessing import Pool
from copy import deepcopy

np.random.seed(42)


def ranking_is_correct(ranking, spines):
    ranking = np.asarray(ranking)
    tasks = len(ranking)
    leaves = len(ranking[0])
    for leaf_hosts in ranking.sum(axis=0):
        if leaf_hosts > spines:
            return False
    return True


def ranking_is_colored(t, l, s, ranking):
    G = nx.Graph()
    leaves_host_number_on_tasks = [
        [(task, ranking[task][leaf]) for task in range(t)] for leaf in range(l)
    ]
    for leaf in leaves_host_number_on_tasks:
        for task_id, size in leaf:
            for i in range(1, size + 1):
                for j in range(i + 1, size + 1):
                    G.add_edge(f"{task_id}_{i}", f"{task_id}_{j}")
        for from_ in range(0, len(leaf)):
            for to_ in range(from_ + 1, len(leaf)):
                task_from, size_from = leaf[from_]
                task_to, size_to = leaf[to_]
                for i in range(1, size_from + 1):
                    for j in range(1, size_to + 1):
                        G.add_edge(f"{task_from}_{i}", f"{task_to}_{j}")

    model = cp_model.CpModel()

    # search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # search_parameters.time_limit.seconds = 300

    node_order = {node: i for i, node in enumerate(G.nodes())}
    variables = [model.NewIntVar(1, s, node) for node in G.nodes()]
    for u, v in G.edges():
        model.add(variables[node_order[u]] != variables[node_order[v]])
    solver = cp_model.CpSolver()
    # solver.parameters.max_time_in_seconds = 300

    status = solver.solve(model)
    return (status == cp_model.OPTIMAL) or (status == cp_model.FEASIBLE)


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

#     solver = pywrapcp.Solver("CPSolver")
#     # search_parameters = pywrapcp.DefaultRoutingSearchParameters()
#     # search_parameters.time_limit.seconds = 300

#     node_order = {node: i for i, node in enumerate(G.nodes())}
#     variables = [solver.IntVar(1, s, node) for node in G.nodes()]
#     for u, v in G.edges():
#         solver.Add(variables[node_order[u]] != variables[node_order[v]])
#     decision_builder = solver.Phase(
#         variables, solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE
#     )

#     solver.NewSearch(decision_builder)
#     if not solver.NextSolution():
#         return False
#     return True


def generate_test_alpha(tasks, leaves, spines, alpha=1, task_distribution=None):
    res = []
    bounds = [spines] * leaves
    for task in range(tasks):
        task_location = [0] * leaves
        task_prob = np.ones(leaves, dtype=np.float64)
        for leaf in range(leaves):
            if bounds[leaf] == 0:
                task_prob[leaf] = 0
        if task_distribution is None:
            two_power_border = int(
                math.log2(1 + min(max(bounds), sum(bounds) - 2 * (tasks - task - 1)))
            )
            two_power = 2 ** np.random.choice(list(range(1, two_power_border + 1)))
        else:
            two_power = task_distribution[task]
        for host in range(two_power):
            if np.sum(task_prob) == 0:
                break
            host_leaf = np.random.choice(
                np.arange(leaves), p=task_prob / np.sum(task_prob)
            )
            task_location[host_leaf] += 1
            bounds[host_leaf] -= 1
            if bounds[host_leaf] > 0:
                task_prob[host_leaf] = math.e ** (alpha * (spines - bounds[host_leaf]))
            else:
                task_prob[host_leaf] = 0
        res.append(task_location)
    return res


def get_task_sizes(tasks, N):
    logN = int(np.log2(N)) + 1
    problem = pulp.LpProblem("task_distribution", pulp.LpMaximize)

    variables = pulp.LpVariable.dict("Var", range(logN), lowBound=0, cat="Integer")

    problem += pulp.lpSum(
        [
            (2**i if i >= math.floor(logN / 2) + 1 else 2 ** (i - logN)) * variables[i]
            for i in range(logN)
        ]
    )
    problem += pulp.lpSum([(2**i) * variables[i] for i in range(logN)]) <= N
    problem += pulp.lpSum([variables[i] for i in range(logN)]) == tasks

    problem.solve()
    res = []
    for i, v in enumerate(problem.variables()):
        if int(v.varValue) == 0:
            continue
        res += [2**i] * int(v.varValue)
    return res


def get_task_sizes_pangu(tasks, N):
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
        <= N
    )
    problem += 1 * x0 + 3 * x1 + 4**x2 + 8 * x3 + 12 * x4 + 24 * x5 == tasks

    problem.solve()

    res = []
    res += [8] * (int(pulp.value(x0)) * 1)
    res += [8] * (int(pulp.value(x1)) * 3)
    res += [8] * (int(pulp.value(x2)) * 4)
    res += [16] * (int(pulp.value(x3)) * 8)
    res += [16] * (int(pulp.value(x4)) * 12)
    res += [16] * (int(pulp.value(x5)) * 24)
    return res


def generate_dense_test_alpha(tasks, leaves, spines, alpha=1):
    task_distribution = get_task_sizes(tasks, leaves * spines)
    return generate_test_alpha(tasks, leaves, spines, alpha, task_distribution)


def generate_tests_alpha(
    tasks, leaves, spines, alpha=1, test_num=10, task_distribution=None
):
    tests = set()

    for _ in range(test_num):
        N = sum(task_distribution)
        diff = 0
        while leaves * (spines - diff) >= N:
            diff += 1
        ranking = generate_test_alpha(
            tasks, leaves, spines - diff + 1, alpha, task_distribution
        )

        if not ranking_is_correct(ranking, spines):
            break
        ranking = sorted(ranking, key=lambda x: sum(x))
        ranking = np.asarray(ranking)
        ranking = ranking[:, ranking.sum(axis=0).argsort()]
        is_colored = ranking_is_colored(tasks, leaves, spines, ranking)
        # tests.add(tuple(map(tuple, ranking)))
        # tests.add(ranking)

        tests.add((tuple([tuple(map(int, row)) for row in ranking]), is_colored))
    return tests


# def generate_test(tasks, leaves, spines):
#     res = []
#     bounds = [spines] * leaves
#     for task in range(tasks):
#         task_distr = [0] * leaves
#         two_power = 2 ** np.random.choice(
#             list(range(1, int(math.log2(1 + leaves * max(bounds)))))
#         )
#         permute = []
#         for leaf in range(leaves):
#             permute.extend([leaf] * (bounds[leaf]))
#         np.random.shuffle(permute)
#         for l in permute[:two_power]:
#             task_distr[l] += 1
#         res.append(task_distr)
#         for l in range(leaves):
#             bounds[l] -= task_distr[l]

#     # res = np.asarray(res)
#     return res


# def generate_tests(tasks, leaves, spines, number=10):
#     tests = set()
#     for _ in range(number):
#         ranking = generate_test(tasks, leaves, spines)
#         ranking = sorted(ranking, key=lambda x: sum(x))
#         ranking = np.asarray(ranking)
#         ranking = ranking[:, ranking.sum(axis=0).argsort()]

#         # tests.add(tuple(map(tuple, ranking)))
#         # tests.add(ranking)
#         tests.add(tuple([tuple(map(int, row)) for row in ranking]))
#     return tests

# lock = threading.Lock()


def write_tests(tasks, leaves, spines, alpha=1, test_num=10, task_distribution=None):
    tests = generate_tests_alpha(
        tasks, leaves, spines, alpha, test_num, task_distribution
    )
    coloring = [x[1] for x in tests]
    tests = [x[0] for x in tests]
    # file_name = f"{tasks}_{leaves}_{spines}"
    file_name = f"{tasks}_{leaves}_{spines}"
    cwd = os.getcwd()
    parent1 = os.path.dirname(cwd)

    test_file_dir = os.path.join(
        parent1, "MCF_spine_leaf", "new_multi_tasks_tests", file_name, str(alpha)
    )
    # with lock:
    if not os.path.exists(test_file_dir):
        os.makedirs(test_file_dir)
    num = 0
    while os.path.isfile(os.path.join(test_file_dir, f"test_{num}.json")):
        num += 1
    test_file = os.path.join(test_file_dir, f"test_{num}.json")
    color_file = os.path.join(test_file_dir, f"color_{num}.json")
    with open(test_file, "a") as f:
        json.dump(list(tests), f)
    with open(color_file, "w") as f:
        json.dump(coloring, f)


def pool_write_tests(x):
    tasks, leaves, spines, alpha, test_num, task_distribution, file_num = x
    tests = generate_tests_alpha(
        tasks, leaves, spines, alpha, test_num, task_distribution
    )
    coloring = [x[1] for x in tests]
    tests = [x[0] for x in tests]
    # file_name = f"{tasks}_{leaves}_{spines}"
    file_name = f"{tasks}_{leaves}_{spines}"
    cwd = os.getcwd()
    parent1 = os.path.dirname(cwd)

    test_file_dir = os.path.join(
        parent1, "MCF_spine_leaf", "new_new_multi_tasks_tests", file_name, str(alpha)
    )
    # if not os.path.exists(test_file_dir):
    os.makedirs(test_file_dir, exist_ok=True)
    test_file = os.path.join(test_file_dir, f"test_{file_num}.json")
    color_file = os.path.join(test_file_dir, f"color_{file_num}.json")
    with open(test_file, "a") as f:
        json.dump(list(tests), f)
    with open(color_file, "w") as f:
        json.dump(coloring, f)


if __name__ == "__main__":
    # t, l, s, alpha = 8, 32, 16, -0.8
    # res = generate_dense_test_alpha(t, l, s, alpha)
    # for x in res:
    #     print(x)
    # print(list(map(int, np.sum(res, axis=0))))
    # print(list(map(int, np.sum(res, axis=1))))
    # print(sum(list(map(int, np.sum(res, axis=1)))), "<=", l * s)

    # t, l, s, alpha = 3, 10, 20, -0.1
    alpha_down, alpha_up = -1, 1
    step = 0.1
    alpha_list = [round(x, 5) for x in np.arange(alpha_down, alpha_up + step, step)]
    alpha_list[10] = np.float64(0.0)
    # old_tests = [
    #     (1, 4, 8, [16]),
    #     (1, 64, 32, [1024]),
    #     (1, 64, 32, [512]),
    #     (1, 16, 8, [32]),
    #     (1, 16, 8, [16]),
    #     (1, 32, 16, [128]),
    #     (1, 32, 16, [64]),
    # ]
    # tests = [(1, 16, 8, [64]), (1, 32, 16, [256])]
    # tests = [(1, 8, 4, [16]), (1, 8, 4, [8])]
    # tests = [(1, 64, 32, [256])]
    tests = [
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
        # [2, 16, 8, [64, 64]],
        # [2, 16, 8, [32, 32]],
        # [2, 16, 8, [8, 32]],
        # [3, 16, 8, [32, 32, 64]],
        # [3, 16, 8, [16, 16, 32]],
        # [3, 16, 8, [8, 16, 16]],
        # [4, 16, 8, [16, 16, 32, 64]],
        # [4, 16, 8, [16, 16, 16, 16]],
        # [4, 16, 8, [8, 8, 8, 16]],
        # [5, 16, 8, [16, 16, 16, 16, 64]],
        # [5, 16, 8, [8, 8, 8, 8, 32]],
        # [5, 16, 8, [8, 8, 8, 8, 8]],
        # [6, 16, 8, [8, 8, 8, 8, 32, 64]],
        # [6, 16, 8, [8, 8, 8, 8, 16, 16]],
        # [6, 16, 8, [4, 4, 8, 8, 8, 8]],
        # [7, 16, 8, [16, 16, 16, 16, 16, 16, 32]],
        # [7, 16, 8, [8, 8, 8, 8, 8, 8, 16]],
        # [7, 16, 8, [4, 4, 4, 4, 8, 8, 8]],
        # [8, 16, 8, [16, 16, 16, 16, 16, 16, 16, 16]],
        # [2, 32, 16, [256, 256]],
        # [2, 8, 4, [8, 16]],
        # [3, 8, 4, [4, 4, 16]],
        # [4, 8, 4, [4, 4, 8, 8]],
        # [5, 8, 4, [4, 4, 4, 4, 8]],
        # [2, 16, 8, [32, 64]],
        # [3, 16, 8, [16, 16, 64]],
        # [4, 16, 8, [8, 8, 16, 64]],
        # [5, 16, 8, [8, 8, 8, 8, 64]],
        # [6, 16, 8, [4, 4, 4, 4, 16, 64]],
        # [7, 16, 8, [4, 4, 8, 8, 8, 32, 32]],
        # [8, 16, 8, [8, 8, 8, 8, 8, 8, 16, 32]],
        # [2, 32, 16, [128, 256]],
        [3, 32, 16, [64, 64, 256]],
        [4, 32, 16, [32, 32, 64, 256]],
        [5, 32, 16, [32, 32, 32, 32, 256]],
        [6, 32, 16, [16, 16, 32, 32, 32, 256]],
        [7, 32, 16, [16, 16, 16, 16, 32, 32, 256]],
        [8, 32, 16, [4, 4, 4, 4, 16, 32, 64, 256]],
        #
        # [2, 32, 16, [128, 128]],
        # [2, 32, 16, [32, 128]],
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
    ]
    new_tests = [
        # [4, 16, 8, [32, 32, 32, 32]],
        # [5, 16, 8, [8, 8, 16, 32, 64]],
        # [5, 16, 8, [16, 16, 32, 32, 32]],
        # [6, 16, 8, [4, 4, 8, 16, 32, 64]],
        # [6, 16, 8, [8, 8, 16, 16, 16, 64]],
        # [6, 16, 8, [8, 8, 16, 32, 32, 32]],
        # [6, 16, 8, [16, 16, 16, 16, 32, 32]],
        # [7, 16, 8, [4, 4, 4, 4, 16, 32, 64]],
        # [7, 16, 8, [4, 4, 8, 16, 16, 16, 64]],
        # [7, 16, 8, [4, 4, 8, 16, 32, 32, 32]],
        # [7, 16, 8, [8, 8, 8, 8, 32, 32, 32]],
        # [7, 16, 8, [8, 8, 8, 8, 16, 16, 64]],
        # [7, 16, 8, [8, 8, 16, 16, 16, 32, 32]],
        # [7, 16, 8, [4, 4, 8, 8, 8, 32, 64]],
        [8, 16, 8, [4, 4, 4, 4, 16, 32, 32, 32], 0],
        [8, 16, 8, [4, 4, 4, 4, 8, 8, 32, 64], 1],
        [8, 16, 8, [4, 4, 4, 4, 16, 16, 16, 64], 2],
        [8, 16, 8, [4, 4, 8, 8, 8, 32, 32, 32], 3],
        [8, 16, 8, [8, 8, 8, 8, 8, 8, 16, 64], 4],
        [8, 16, 8, [8, 8, 8, 8, 16, 16, 32, 32], 5],
        [8, 16, 8, [8, 8, 16, 16, 16, 16, 16, 32], 6],
        [8, 16, 8, [4, 4, 8, 16, 16, 16, 32, 32], 7],
        [8, 16, 8, [4, 4, 8, 8, 8, 16, 16, 64], 8],
        #
        [2, 16, 4, [32, 32], 0],
        [2, 16, 4, [16, 32], 1],
        [3, 16, 4, [16, 16, 32], 0],
        [3, 16, 4, [4, 16, 32], 1],
        [3, 16, 4, [8, 8, 32], 2],
        [4, 16, 4, [8, 8, 16, 32], 0],
        [4, 16, 4, [4, 8, 8, 32], 1],
        [4, 16, 4, [4, 4, 8, 32], 2],
        [4, 16, 4, [16, 16, 16, 16], 3],
        [4, 16, 4, [4, 16, 16, 16], 4],
        [4, 16, 4, [8, 8, 16, 16], 5],
        [5, 16, 4, [4, 4, 8, 16, 32], 0],
        [5, 16, 4, [4, 4, 4, 8, 32], 1],
        [5, 16, 4, [4, 4, 4, 4, 32], 2],
        [5, 16, 4, [8, 8, 16, 16, 16], 3],
        [5, 16, 4, [8, 8, 8, 8, 32], 4],
        [5, 16, 4, [4, 8, 8, 16, 16], 5],
        [5, 16, 4, [8, 8, 8, 8, 16], 6],
        [6, 16, 4, [4, 4, 4, 4, 16, 32], 0],
        [6, 16, 4, [4, 4, 4, 4, 4, 32], 1],
        [6, 16, 4, [8, 8, 8, 8, 8, 8], 2],
        [6, 16, 4, [4, 4, 8, 16, 16, 16], 3],
        [6, 16, 4, [8, 8, 8, 8, 16, 16], 4],
        [6, 16, 4, [4, 4, 8, 8, 8, 32], 5],
        [6, 16, 4, [4, 4, 4, 8, 16, 16], 6],
        [6, 16, 4, [4, 8, 8, 8, 8, 16], 7],
        [6, 16, 4, [4, 4, 8, 8, 8, 16], 8],
        [6, 16, 4, [4, 4, 4, 4, 16, 16], 9],
        [7, 16, 4, [4, 4, 4, 4, 16, 16, 16], 0],
        [7, 16, 4, [4, 8, 8, 8, 8, 8, 8], 1],
        [7, 16, 4, [4, 4, 8, 8, 8, 8, 8], 2],
        [7, 16, 4, [4, 4, 8, 8, 8, 16, 16], 3],
        [7, 16, 4, [4, 4, 4, 4, 8, 8, 32], 4],
        [7, 16, 4, [8, 8, 8, 8, 8, 8, 16], 5],
        [7, 16, 4, [4, 4, 4, 8, 8, 8, 16], 6],
        [7, 16, 4, [4, 4, 4, 4, 4, 16, 16], 7],
        [7, 16, 4, [4, 4, 4, 4, 8, 8, 16], 8],
        [8, 16, 4, [4, 4, 4, 4, 4, 4, 8, 32], 0],
        [8, 16, 4, [4, 4, 4, 8, 8, 8, 8, 8], 1],
        [8, 16, 4, [4, 4, 4, 4, 8, 8, 8, 8], 2],
        [8, 16, 4, [8, 8, 8, 8, 8, 8, 8, 8], 3],
        [8, 16, 4, [4, 4, 4, 4, 8, 8, 16, 16], 4],
        [8, 16, 4, [4, 4, 8, 8, 8, 8, 8, 16], 5],
        [8, 16, 4, [4, 4, 4, 4, 4, 8, 8, 16], 6],
        [8, 16, 4, [4, 4, 4, 4, 4, 4, 8, 16], 7],
        [2, 32, 8, [128, 128], 0],
        [2, 32, 8, [64, 128], 1],
        [3, 32, 8, [64, 64, 128], 0],
        [3, 32, 8, [16, 64, 128], 1],
        [3, 32, 8, [32, 32, 128], 2],
        [3, 32, 8, [64, 64, 64], 3],
        [4, 32, 8, [64, 64, 64, 64], 0],
        [4, 32, 8, [8, 16, 64, 128], 1],
        [4, 32, 8, [16, 16, 32, 128], 2],
        [4, 32, 8, [32, 32, 64, 64], 3],
        [4, 32, 8, [32, 32, 64, 128], 4],
        [5, 32, 8, [32, 32, 64, 64, 64], 0],
        [5, 32, 8, [8, 8, 32, 32, 128], 1],
        [5, 32, 8, [4, 8, 16, 64, 128], 2],
        [5, 32, 8, [4, 8, 16, 32, 128], 3],
        [5, 32, 8, [16, 16, 32, 64, 64], 4],
        [5, 32, 8, [32, 32, 32, 32, 128], 5],
        [5, 32, 8, [16, 16, 32, 64, 128], 6],
        [5, 32, 8, [8, 8, 16, 32, 128], 7],
        [5, 32, 8, [32, 32, 32, 32, 64], 8],
        [5, 32, 8, [16, 16, 16, 16, 128], 9],
        [6, 32, 8, [4, 4, 32, 64, 64, 64], 0],
        [6, 32, 8, [32, 32, 32, 32, 64, 64], 1],
        [6, 32, 8, [4, 4, 4, 16, 64, 128], 2],
        [6, 32, 8, [4, 4, 8, 16, 32, 128], 3],
        [6, 32, 8, [16, 16, 32, 64, 64, 64], 0],
        [6, 32, 8, [16, 16, 16, 16, 64, 128], 1],
        [6, 32, 8, [16, 16, 32, 32, 32, 128], 3],
        [6, 32, 8, [8, 8, 16, 32, 64, 128], 4],
        [6, 32, 8, [4, 8, 16, 32, 32, 128], 5],
        [6, 32, 8, [4, 8, 8, 8, 64, 128], 6],
        [6, 32, 8, [4, 8, 16, 64, 64, 64], 7],
        # [6, 32, 8, [8, 8, 8, 8, 32, 128], 9],
        # [6, 32, 8, [32, 32, 32, 32, 32, 32], 10],
        # [6, 32, 8, [16, 16, 16, 16, 64, 64], 11],
        # [6, 32, 8, [16, 16, 32, 32, 32, 64], 13],
        # [6, 32, 8, [8, 8, 16, 32, 64, 64], 14],
        # [6, 32, 8, [8, 8, 16, 16, 16, 128], 15],
        [7, 32, 8, [4, 4, 4, 32, 64, 64, 64], 0],
        [7, 32, 8, [16, 16, 16, 16, 64, 64, 64], 1],
        [7, 32, 8, [4, 4, 4, 8, 8, 64, 128], 2],
        [7, 32, 8, [4, 4, 4, 4, 16, 32, 128], 3],
        [7, 32, 8, [8, 8, 8, 8, 32, 64, 128], 4],
        [7, 32, 8, [32, 32, 32, 32, 32, 32, 64], 5],
        [7, 32, 8, [8, 8, 16, 16, 16, 64, 128], 6],
        [7, 32, 8, [4, 4, 8, 16, 32, 64, 128], 7],
        [7, 32, 8, [16, 16, 16, 16, 32, 32, 128], 8],
        [7, 32, 8, [8, 8, 16, 32, 64, 64, 64], 9],
        # [7, 32, 8, [4, 4, 4, 16, 64, 64, 64], 10],
        # [7, 32, 8, [4, 8, 8, 8, 64, 64, 64], 11],
        # [7, 32, 8, [4, 8, 16, 32, 32, 64, 64], 12],
        # [7, 32, 8, [4, 4, 4, 16, 32, 32, 128], 13],
        # [7, 32, 8, [4, 8, 8, 8, 32, 32, 128], 14],
        # [7, 32, 8, [4, 8, 16, 16, 16, 32, 128], 15],
        # [7, 32, 8, [16, 16, 32, 32, 32, 32, 32], 16],
        # [7, 32, 8, [16, 16, 32, 32, 32, 64, 64], 17],
        # [7, 32, 8, [4, 4, 8, 16, 32, 64, 64], 18],
        # [7, 32, 8, [8, 8, 8, 8, 16, 16, 128], 19],
        # [7, 32, 8, [4, 4, 8, 16, 16, 16, 128], 20],
        # [7, 32, 8, [8, 8, 16, 32, 32, 32, 64], 21],
        # [7, 32, 8, [8, 8, 16, 16, 16, 64, 64], 22],
        # [7, 32, 8, [8, 8, 8, 8, 32, 64, 64], 23],
        # [7, 32, 8, [16, 16, 16, 16, 32, 32, 64], 24],
        # [7, 32, 8, [4, 4, 8, 8, 8, 32, 128], 25],
        # [7, 32, 8, [8, 8, 16, 32, 32, 32, 128], 26],
        [8, 32, 8, [4, 4, 4, 4, 32, 64, 64, 64], 0],
        [8, 32, 8, [8, 8, 16, 16, 16, 64, 64, 64], 1],
        [8, 32, 8, [8, 8, 8, 8, 16, 16, 64, 128], 2],
        [8, 32, 8, [4, 4, 8, 16, 16, 16, 64, 128], 3],
        # [9, 32, 8, [4, 4, 4, 4, 4, 32, 64, 64, 64], 0],
        # [9, 32, 8, [16, 16, 16, 16, 16, 16, 16, 16, 128], 1],
        # [9, 32, 8, [4, 4, 4, 4, 4, 4, 4, 64, 128], 2],
        # [9, 32, 8, [4, 4, 4, 4, 4, 4, 8, 32, 128], 3],
    ]
    new_new_tests = [
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
        [5, 16, 8, [8, 8, 16, 32, 32], 3],
        [5, 16, 8, [4, 4, 8, 16, 64], 4],
        [5, 16, 8, [8, 8, 8, 8, 64], 5],
        [5, 16, 8, [8, 8, 16, 16, 32], 6],
        [5, 16, 8, [16, 16, 16, 16, 16], 7],
        [6, 16, 8, [4, 4, 8, 16, 16, 64], 0],
        [6, 16, 8, [4, 4, 8, 32, 32, 32], 1],
        [6, 16, 8, [16, 16, 16, 16, 16, 32], 2],
        [6, 16, 8, [8, 8, 16, 16, 32, 32], 3],
        [6, 16, 8, [8, 8, 8, 8, 16, 64], 4],
        [6, 16, 8, [8, 8, 8, 8, 32, 32], 5],
        [6, 16, 8, [4, 4, 8, 16, 32, 32], 6],
        [6, 16, 8, [16, 16, 16, 16, 16, 16], 7],
        [6, 16, 8, [4, 4, 4, 4, 16, 64], 8],
        [6, 16, 8, [8, 8, 8, 8, 16, 32], 9],
        [6, 16, 8, [4, 4, 4, 4, 32, 32], 10],
        [7, 16, 8, [8, 8, 8, 8, 16, 32, 32], 0],
        [7, 16, 8, [4, 4, 8, 8, 8, 16, 64], 1],
        [7, 16, 8, [8, 8, 8, 8, 8, 8, 64], 2],
        [7, 16, 8, [4, 4, 4, 4, 16, 16, 64], 3],
        [7, 16, 8, [8, 8, 8, 8, 16, 16, 32], 4],
        [7, 16, 8, [4, 4, 4, 4, 16, 32, 32], 5],
        [7, 16, 8, [4, 4, 4, 4, 8, 8, 64], 6],
        [7, 16, 8, [4, 4, 8, 8, 8, 32, 32], 7],
        [7, 16, 8, [4, 4, 8, 16, 16, 16, 16], 8],
        [7, 16, 8, [4, 4, 8, 8, 8, 16, 32], 9],
        [7, 16, 8, [4, 4, 4, 4, 16, 16, 32], 10],
        # [8, 16, 8, [4, 4, 8, 8, 8, 8, 8, 64], 0],
        # [8, 16, 8, [8, 8, 8, 8, 8, 8, 32, 32], 1],
        # [8, 16, 8, [4, 4, 8, 16, 16, 16, 16, 32], 2],
        # [8, 16, 8, [8, 8, 8, 8, 16, 16, 16, 32], 3],
        # [8, 16, 8, [8, 8, 16, 16, 16, 16, 16, 16], 4],
        # [8, 16, 8, [4, 4, 4, 4, 16, 16, 32, 32], 5],
        # [8, 16, 8, [4, 4, 4, 4, 16, 16, 16, 32], 6],
        # [8, 16, 8, [4, 4, 4, 4, 8, 8, 32, 32], 7],
        # [8, 16, 8, [4, 4, 4, 4, 4, 4, 8, 64], 8],
        # [8, 16, 8, [4, 4, 8, 16, 16, 16, 16, 16], 9],
        # [8, 16, 8, [4, 4, 4, 4, 16, 16, 16, 16], 10],
        # [8, 16, 8, [4, 4, 8, 8, 8, 8, 8, 32], 11],
        # [8, 16, 8, [8, 8, 8, 8, 8, 8, 16, 16], 12],
        # [8, 16, 8, [4, 4, 8, 8, 8, 16, 16, 16], 13],
    ]
    not_yet_done = []
    args_for_pool = []
    for t, l, s, task_distr, file_num in new_new_tests:
        # task_distribution = get_task_sizes(t, l * s)
        for alpha in alpha_list:
            args_for_pool.append((t, l, s, alpha, 1000, task_distr, file_num))

    not_done_args = []
    done_args = []
    for x in args_for_pool:
        # print(x)
        tasks, leaves, spines, alpha, test_num, task_distribution, file_num = x

        file_name = f"{tasks}_{leaves}_{spines}"
        cwd = os.getcwd()
        parent1 = os.path.dirname(cwd)

        test_file_dir = os.path.join(
            parent1,
            "MCF_spine_leaf",
            "new_new_multi_tasks_tests",
            file_name,
            str(alpha),
        )
        test_file = os.path.join(test_file_dir, f"test_{file_num}.json")
        # color_file = os.path.join(test_file_dir, f"color_{file_num}.json")
        if not os.path.exists(test_file):
            not_done_args.append(x)
        else:
            done_args.append(x)
    print("NOT DONE ARGUMENTS:")
    for x in not_done_args:
        print(x)
    print(len(not_done_args), "NOT DONE ARGUMENTS", len(done_args)+len(not_done_args), "TOTAL ARGUMENTS")
    # print("\n\n-----------\n\n")
    # not_done_args = [x for x in not_done_args if x[1] == 16]
    # print(len(not_done_args), "NOT DONE ARGUMENTS:")
    # not_done_args = [x for x in not_done_args if x[1] == 16]
    # print("DONE ARGUMENTS:")
    # for x in done_args:
    #     print(x)

    # with Pool(60) as p:
    #     p.map(pool_write_tests, not_done_args)
