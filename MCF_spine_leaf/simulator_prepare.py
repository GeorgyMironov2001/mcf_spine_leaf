import os
import json
import numpy as np
from pathlib import Path
import networkx as nx


class Test:
    def __init__(
        self,
        t,
        l,
        s,
        ranking,
        layer_colors,
        communication_colors,
        final_ranking,
        result,
    ):
        self.tasks = t
        self.leaves = l
        self.spines = s
        self.ranking = ranking
        self.layer_colors = layer_colors
        self.communication_colors = communication_colors
        self.final_ranking = final_ranking
        self.result = result


def parse_coloring(colors_list):
    res = {}
    for (task, stage, rank_from, rank_to), size in colors_list:
        if task not in res:
            res[task] = {}
        if stage not in res[task]:
            res[task][stage] = []
        res[task][stage].append([rank_from, rank_to, size])
    return res


def read_test(name, test_dir_name, result_dir_name):
    tasks, leaves, spines = list(map(int, name.split("_")))

    res_tests = []
    test_dir_path = Path(test_dir_name) / name
    res_dir_path = Path(result_dir_name) / name
    count = int(
        len(
            [
                entry
                for entry in os.listdir(test_dir_path)
                if os.path.isfile(os.path.join(test_dir_path, entry))
            ]
        )
        / 2
    )
    for num in range(count):
        test_data = None
        test_color = None

        res_coloring = None
        res_ranking = None
        res_result = None
        with open(os.path.join(test_dir_path, f"test_{num}.json"), "r") as f:
            test_data = json.load(f)
        with open(os.path.join(test_dir_path, f"color_{num}.json"), "r") as f:
            test_color = json.load(f)

        with open(os.path.join(res_dir_path, f"coloring_{num}.json"), "r") as f:
            res_coloring = json.load(f)
        with open(os.path.join(res_dir_path, f"ranking_{num}.json"), "r") as f:
            res_ranking = json.load(f)
        with open(os.path.join(res_dir_path, f"test_{num}.json"), "r") as f:
            res_result = json.load(f)

        res_tests += [
            Test(
                tasks,
                leaves,
                spines,
                test_data[i],
                test_color[i],
                parse_coloring(res_coloring[i]),
                res_ranking[i],
                res_result[i],
            )
            for i in range(len(test_color))
        ]
    return res_tests


def finish_broken_test(test: Test):
    additional_colors = []

    for task in range(test.tasks):
        hosts = sum(test.ranking[task])

        host_to_leaf = [-1] * hosts
        for leaf_id, leaf in enumerate(test.final_ranking[task]):
            for host in leaf:
                host_to_leaf[host] = leaf_id

        stages = int(np.log2(hosts))
        for stage in range(stages):
            edge_dict = {}

            add_power = 2**stage
            G = nx.MultiGraph()
            for rank_from in range(hosts):
                rank_to = rank_from ^ add_power
                if host_to_leaf[rank_from] == host_to_leaf[rank_to]:
                    continue
                edge_dict[(rank_from, rank_to)] = G.add_edge(
                    f"{host_to_leaf[rank_from]}_s",
                    f"{host_to_leaf[rank_to]}_d",
                    rank_s=rank_from,
                    rank_d=rank_to,
                    color=-1,
                )

            if not (
                task in test.communication_colors
                and stage in test.communication_colors[task]
            ):
                continue
            for rank_from, rank_to, size in test.communication_colors[task][stage]:
                G[f"{host_to_leaf[rank_from]}_s"][f"{host_to_leaf[rank_to]}_d"][
                    edge_dict[(rank_from, rank_to)]
                ]["color"] = size

            for rank_from in range(hosts):
                rank_to = rank_from ^ add_power
                if host_to_leaf[rank_from] == host_to_leaf[rank_to]:
                    continue
                leaf_s = f"{host_to_leaf[rank_from]}_s"
                leaf_d = f"{host_to_leaf[rank_to]}_d"
                key = edge_dict[(rank_from, rank_to)]
                if G[leaf_s][leaf_d][key]["color"] != -1:
                    continue
                available_colors = min(
                    test.ranking[task][host_to_leaf[rank_from]],
                    test.ranking[task][host_to_leaf[rank_to]],
                )
                used_colors = {c: 0 for c in range(1, available_colors + 1)}

                for leaf in [leaf_s, leaf_d]:
                    for leaf_nbr in G.neighbors(leaf):
                        for edge_key, color_dict in G[leaf][leaf_nbr].items():
                            if (
                                color_dict["color"] != -1
                                and color_dict["color"] <= available_colors
                            ):
                                used_colors[color_dict["color"]] += 1
                min_color, min_times = min(used_colors.items(), key=lambda x: x[1])
                G[leaf_s][leaf_d][key]["color"] = min_color
                additional_colors.append([[task, stage, rank_from, rank_to], min_color])
    return additional_colors


def update_broken_test(test: Test):
    additional_colors = finish_broken_test(test)
    for (task, stage, rank_from, rank_to), size in additional_colors:
        if task not in test.communication_colors:
            test.communication_colors[task] = {}
        if stage not in test.communication_colors[task]:
            test.communication_colors[task][stage] = []
        test.communication_colors[task][stage].append([rank_from, rank_to, size])


def get_coloring(test: Test):
    res = []
    for task, task_dict in test.communication_colors.items():
        for stage, vec in task_dict.items():
            for rank_from, rank_to, size in vec:
                res.append(
                    [
                        [task, stage, rank_from, rank_to],
                        test.layer_colors[f"{task}_{size}"],
                    ]
                )
    return res


if __name__ == "__main__":

    t_l_s = [[2, 16, 8], [3, 16, 8], [7, 16, 8]]
    for t, l, s in t_l_s:
        file_name = f"{t}_{l}_{s}"
        output_path = (
            Path(os.getcwd()) / "new_special_multitests_scenario2_simulator_data" / file_name
        )
        if os.path.exists(output_path):
            continue
        tests = read_test(
            file_name, "new_special_multitests", "new_special_multitests_scenario2"
        )
        for test in tests:
            if test.result != 0:
                continue
            update_broken_test(test)

        final_rankings = list([test.final_ranking for test in tests])
        colorings = list([get_coloring(test) for test in tests])
        results = list([test.result for test in tests])

        os.makedirs(output_path, exist_ok=True)

        ranking_file = output_path / "rankings.json"
        coloring_file = output_path / "colorings.json"
        result_file = output_path / "results.json"

        for file_data, data in [
            (ranking_file, final_rankings),
            (coloring_file, colorings),
            (result_file, results),
        ]:
            print(file_data)
            with open(file_data, "w") as f:
                f.truncate(0)
                json.dump(data, f)
