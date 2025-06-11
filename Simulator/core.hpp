#pragma once

#include "Task.hpp"
#include <algorithm>
#include <memory>
#include <queue>
#include <random>

struct State {
  std::vector<Task> tasks;
  std::vector<std::vector<std::queue<Data>>> input_queues;
  std::vector<std::vector<std::queue<Data>>> output_queues;
  State() = default;
  State(const std::vector<Task> &tasks, int spines, int leaves) : tasks(tasks) {
    input_queues.resize(leaves, std::vector<std::queue<Data>>(spines));
    output_queues.resize(leaves, std::vector<std::queue<Data>>(spines));
  }
};

struct Decider {

  Decider() = default;
  virtual void decide(Data data, State &state) {};
  virtual ~Decider() = default;
};

struct Simulator {
  int spines;
  int leaves;
  State state;
  std::unique_ptr<Decider> decider;
  int step_counter;
  bool need_stop;

  std::mt19937 rng;

  Simulator(int spines, int leaves, const State &state, std::mt19937 rng)
      : spines(spines), leaves(leaves), state(state), rng(rng) {
    step_counter = 0;
    need_stop = false;
  }

  void step() {
    step_counter += 1;

    for (int leaf_id = 0; leaf_id < leaves; ++leaf_id) {
      for (int spine_id = 0; spine_id < spines; ++spine_id) {

        auto &leaf_spine_queue_input = state.input_queues[leaf_id][spine_id];
        if (leaf_spine_queue_input.empty()) {
          continue;
        }
        auto data = leaf_spine_queue_input.front();
        auto [task_id, rank_from, rank_to] = data;
        leaf_spine_queue_input.pop();
        int leaf_to = state.tasks[task_id].host_to_leaf[rank_to];

        state.output_queues[leaf_to][spine_id].push(data);
      }
    }

    for (int leaf_id = 0; leaf_id < leaves; ++leaf_id) {
      for (int spine_id = 0; spine_id < spines; ++spine_id) {

        auto &leaf_spine_queue_output = state.output_queues[leaf_id][spine_id];
        if (leaf_spine_queue_output.empty()) {
          continue;
        }
        auto [task_id, rank_from, rank_to] = leaf_spine_queue_output.front();
        leaf_spine_queue_output.pop();
        state.tasks[task_id].stage_counter -= 1;
      }
    }
    //   if (spine_queue.empty()) {
    //     continue;
    //   }
    //   auto [task_id, rank_from, rank_to] = spine_queue.front();
    //   spine_queue.pop();
    //   tasks[task_id].stage_counter -= 1;

    int finished = 0;
    std::vector<Data> new_stages_data;

    for (int task_id = 0; task_id < state.tasks.size(); ++task_id) {
      Task &task = state.tasks[task_id];
      if (task.is_finished()) {
        finished += 1;
        continue;
      }
      if (!task.is_starting_next_stage()) {
        continue;
      }
      auto vec = task.start_next_stage();
      new_stages_data.insert(new_stages_data.end(), vec.begin(), vec.end());
    }

    if (finished == state.tasks.size()) {
      need_stop = true;
      return;
    }
    if (new_stages_data.empty()) {
      return;
    }

    std::shuffle(new_stages_data.begin(), new_stages_data.end(), rng);
    for (auto &data : new_stages_data) {
      decider->decide(data, state);
    }
  }

  int do_simulation() {
    while (!need_stop) {
      step();
    }
    return step_counter;
  }
};
