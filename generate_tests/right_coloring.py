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
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.rcsetup as rcsetup

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

    nx.draw_networkx(G, with_labels=True)
    plt.show()

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


def generate_tests_alpha(
    tasks, leaves, spines, alpha=1, test_num=1, task_distribution=None
):
    ranking = generate_test_alpha(tasks, leaves, spines, alpha, task_distribution)

    if not ranking_is_correct(ranking, spines):
        return None, None
    ranking = sorted(ranking, key=lambda x: sum(x))
    ranking = np.asarray(ranking)
    ranking = ranking[:, ranking.sum(axis=0).argsort()]
    is_colored = ranking_is_colored(tasks, leaves, spines, ranking)
    return ranking, is_colored


# np.random.seed(1)
# ranking, is_colored = generate_tests_alpha(
#     4, 16, 8, alpha=-0.9, test_num=1, task_distribution=[16, 16, 32, 64]
# )

# print(ranking)
# print(is_colored)

# print(matplotlib.matplotlib_fname())
plt.plot(range(20), range(20))
plt.show()

# print(rcsetup.all_backends)