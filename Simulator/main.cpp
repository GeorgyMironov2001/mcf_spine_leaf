#include "core.hpp"
#include "deciders/ecmp.hpp"
#include "deciders/fixed.hpp"
#include "deciders/min_queue.hpp"
#include "deciders/rand.hpp"
#include "single_include/nlohmann/json.hpp"
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>

using json = nlohmann::json;

auto get_start_ranking(int tasks, int leaves, int spines,
                       std::vector<std::vector<std::set<int>>> &final_ranking) {
  std::vector<std::vector<std::set<int>>> result_start_ranking(
      tasks, std::vector<std::set<int>>(leaves));
  for (int task_id = 0; task_id < tasks; ++task_id) {
    int ptr = 0;
    for (int leaf_id = 0; leaf_id < leaves; ++leaf_id) {
      int task_leaf_hosts = final_ranking[task_id][leaf_id].size();
      for (int i = 0; i < task_leaf_hosts; ++i) {
        result_start_ranking[task_id][leaf_id].insert(ptr);
        ptr++;
      }
    }
  }
  return result_start_ranking;
}

std::vector<Task>
get_tasks_vector(int tasks, std::vector<std::vector<std::set<int>>> &ranking) {
  std::vector<Task> res;
  for (int task_id = 0; task_id < tasks; ++task_id) {
    res.emplace_back(task_id, ranking[task_id]);
  }
  return res;
}
std::vector<std::tuple<State, State,
                       std::map<std::tuple<int, int, int, int>, int>, int>>
get_simulator_data(int tasks, int leaves, int spines,
                   std::filesystem::path tests_file_dir) {
  std::vector<std::tuple<State, State,
                         std::map<std::tuple<int, int, int, int>, int>, int>>
      simulator_data;

  auto colorings_file = tests_file_dir;
  colorings_file.append("colorings.json");

  auto rankings_file = tests_file_dir;
  rankings_file.append("rankings.json");

  auto results_file = tests_file_dir;
  results_file.append("results.json");

  std::ifstream f(colorings_file);
  json test_colorings = json::parse(f);
  f.close();

  f = std::ifstream(rankings_file);
  json test_rankings = json::parse(f);
  f.close();

  f = std::ifstream(results_file);
  json test_results = json::parse(f);
  f.close();

  int tests_size = test_results.size();

  for (int test_num = 0; test_num < tests_size; ++test_num) {
    auto test_ranking = test_rankings[test_num];
    std::vector<std::vector<std::set<int>>> final_ranking(
        tasks, std::vector<std::set<int>>(leaves));

    for (int task_id = 0; task_id < tasks; ++task_id) {
      auto task_ranking = test_ranking[task_id];
      for (int leaf_id = 0; leaf_id < leaves; ++leaf_id) {
        for (int host_rank : task_ranking[leaf_id]) {
          final_ranking[task_id][leaf_id].insert(host_rank);
        }
      }
    }

    auto start_ranking =
        get_start_ranking(tasks, leaves, spines, final_ranking);
    auto start_tasks = get_tasks_vector(tasks, start_ranking);
    auto final_tasks = get_tasks_vector(tasks, final_ranking);

    auto start_state = State(start_tasks, spines, leaves);
    auto final_state = State(final_tasks, spines, leaves);

    std::map<std::tuple<int, int, int, int>, int> optimal_storage;
    auto test_coloring = test_colorings[test_num];
    for (auto &task_stage_color : test_coloring) {
      int task = task_stage_color[0][0];
      int stage = task_stage_color[0][1];
      int rank_from = task_stage_color[0][2];
      int rank_to = task_stage_color[0][3];
      int spine = task_stage_color[1];
      optimal_storage[std::make_tuple(task, stage, rank_from, rank_to)] = spine;
    }

    int result = test_results[test_num];
    simulator_data.push_back(
        std::make_tuple(start_state, final_state, optimal_storage, result));
  }
  return simulator_data;
}

int main() {
  const int seed = 5;
  std::mt19937 rng(seed);
  std::filesystem::path file_path =
      std::filesystem::current_path()
          .parent_path()
          .parent_path()
          .append("MCF_spine_leaf")
          .append("new_special_multitests_scenario2_simulator_data")
          .append("7_16_8");
  auto data = get_simulator_data(7, 16, 8, file_path);

  for (auto &[start_state, final_state, optimal_storage, res] : data) {
    Simulator sim_ecmp(8, 16, start_state, rng);
    sim_ecmp.decider = std::make_unique<Ecmp>(start_state);
    int ecmp_time = sim_ecmp.do_simulation();

    Simulator sim_scenario_2(8, 16, final_state, rng);
    sim_scenario_2.decider = std::make_unique<Fixed>(optimal_storage);
    int scenario_2_time = sim_scenario_2.do_simulation();
    std::cout << std::format(
                     "scenario 2 result {}, ecmp time {}, scenario 2 time {}",
                     res, ecmp_time, scenario_2_time)
              << std::endl;
  }

  // Simulator sim(32, 64, final_state, rng);
  // sim.decider = std::make_unique<Fixed>(optimal_storage_14_64_32);

  //   Simulator sim(32, 64, start_random_state, rng);
  //   sim.decider = std::make_unique<Ecmp>(start_random_state);

  //   Simulator sim(32, 64, start_random_state, rng);
  //   sim.decider = std::make_unique<Min_queue>();

  //   Simulator sim(32, 64, start_random_state, rng);
  //   sim.decider = std::make_unique<Rand>(3);

  //   std::cout << sim.do_simulation();
}