import json
import os
import matplotlib.pyplot as plt
import numpy as np


def get_statistic(stat_name):
    cwd = os.getcwd()
    with open(os.path.join(cwd, "statistics", stat_name), "r") as f:
        res = json.load(f)
    return res


def draw_success_part(stat_name):
    stats = get_statistic(stat_name)
    n = len(stats)
    fig, axs = plt.subplots(n, figsize=(10, 8))

    for plot_id, (config, results) in enumerate(stats.items()):
        tasks, leaves, spines = eval(config)
        x = list(map(float, results.keys()))
        y = [
            d["passed_tests"] / (d["passed_tests"] + d["failed_tests"])
            for d in results.values()
        ]
        axs[plot_id].scatter(x, y)
        axs[plot_id].set_title(f"{tasks}_{leaves}_{spines}")
        axs[plot_id].set_xlabel("alpha")
        axs[plot_id].set_ylabel("success rate")
    fig.tight_layout()
    plt.show()


draw_success_part("new_scenario2.json")

