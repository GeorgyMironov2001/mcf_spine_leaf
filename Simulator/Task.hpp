#pragma once

#include "data.hpp"
#include <cmath>
#include <set>
#include <vector>

struct Task {
  int hosts;
  int id;
  std::vector<std::set<int>> ranking;
  std::vector<int> host_to_leaf;
  int curr_stage;
  int stages;
  int stage_counter;

  Task(int id, const std::vector<std::set<int>> &ranking)
      : id(id), ranking(ranking) {
    hosts = 0;
    for (auto &leaf : ranking) {
      hosts += leaf.size();
    }
    curr_stage = -1;
    stages = log2(hosts);
    stage_counter = 0;
    host_to_leaf.resize(hosts, -1);
    for (int leaf_id = 0; leaf_id < ranking.size(); ++leaf_id) {
      //   hosts += ranking[leaf_id].size();
      for (int rank : ranking[leaf_id]) {
        host_to_leaf[rank] = leaf_id;
      }
    }
  }

  bool is_finished() {
    return (stage_counter == 0) && (curr_stage == stages - 1);
  }
  bool is_starting_next_stage() {
    return (stage_counter == 0) && (curr_stage < stages - 1);
  }
  std::vector<Data> start_next_stage() {
    std::vector<Data> res;
    curr_stage += 1;
    int two_power = pow(2, curr_stage);
    for (int host_from = 0; host_from < hosts; ++host_from) {
      int host_to = host_from ^ two_power;
      if (host_to_leaf[host_from] == host_to_leaf[host_to]) {
        continue;
      }
      res.emplace_back(id, host_to, host_from);
      stage_counter += 1;
    }
    return res;
  }
};
