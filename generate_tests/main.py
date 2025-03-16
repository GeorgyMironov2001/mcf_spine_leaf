import numpy as np
import random
import math
import json
import os
from pathlib import Path

np.random.seed(42)


def ranking_is_correct(ranking, spines):
    ranking = np.asarray(ranking)
    tasks = len(ranking)
    leaves = len(ranking[0])
    for leaf_hosts in ranking.sum(axis=0):
        if leaf_hosts > spines:
            return False
    return True


def generate_test_alpha(tasks, leaves, spines, alpha=1):
    res = []
    bounds = [spines] * leaves
    for task in range(tasks):
        task_location = [0] * leaves
        task_prob = np.ones(leaves, dtype=np.float64)

        two_power_border = int(
            math.log2(1 + min(max(bounds), sum(bounds) - 2 * (tasks - task - 1)))
        )
        two_power = 2 ** np.random.choice(list(range(1, two_power_border + 1)))
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


def generate_tests_alpha(tasks, leaves, spines, alpha=1, test_num=10):
    tests = set()
    for _ in range(test_num):
        ranking = generate_test_alpha(tasks, leaves, spines, alpha)
        if not ranking_is_correct(ranking, spines):
            break
        ranking = sorted(ranking, key=lambda x: sum(x))
        ranking = np.asarray(ranking)
        ranking = ranking[:, ranking.sum(axis=0).argsort()]

        # tests.add(tuple(map(tuple, ranking)))
        # tests.add(ranking)

        tests.add(tuple([tuple(map(int, row)) for row in ranking]))
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


def write_tests(tasks, leaves, spines, alpha=1, test_num=10):
    tests = generate_tests_alpha(tasks, leaves, spines, alpha, test_num)
    file_name = f"{tasks}_{leaves}_{spines}"
    cwd = os.getcwd()
    parent1 = os.path.dirname(cwd)
    parent2 = os.path.dirname(parent1)

    test_file_dir = os.path.join(
        parent2, "C++Projects", "MCF_spine_leaf", "tests", file_name, str(alpha)
    )

    if not os.path.exists(test_file_dir):
        os.makedirs(test_file_dir)
    num = 0
    while os.path.isfile(os.path.join(test_file_dir, f"test_{num}.json")):
        num += 1
    test_file = os.path.join(test_file_dir, f"test_{num}.json")

    with open(test_file, "a") as f:
        json.dump(list(tests), f)


if __name__ == "__main__":

    # t, l, s, alpha = 3, 10, 20, -0.1
    alpha_down, alpha_up = -1, 1

    step = 0.1
    alpha_list = [round(x, 5) for x in np.arange(alpha_down, alpha_up + step, step)]
    for t, l, s in [(3, 10, 20), (3, 5, 10), (10, 20, 30), (5, 10, 25)]:
        for alpha in alpha_list:
            for _ in range(10):
                write_tests(t, l, s, alpha, 1000)
