import os
from functools import cmp_to_key

l = []

for a in os.listdir("/home/gera/mcf_spine_leaf/MCF_spine_leaf/multi_tasks_tests"):
    l.append(list(map(int, a.split("_"))))


def comp(x, y):
    if (x[1] / x[2]) != (y[1] / y[2]):
        return (x[1] / x[2]) < (y[1] / y[2])
    return x[0] < y[0]


l = sorted(l, key=lambda x: (x[1] / x[2], (x[1], x[2]), x[0]))
for x in l:
    print("{" + f"{x[0]}, {x[1]}, {x[2]}" + "},")
