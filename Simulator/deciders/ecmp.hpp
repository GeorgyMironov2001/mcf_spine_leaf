#include "../core.hpp"
#include <map>
#include <tuple>

struct Ecmp : public Decider {
  std::map<std::tuple<int, int, int, int>, int> storage;

  void decide(Data data, State &state) override {
    auto [task_id, rank_from, rank_to] = data;
    int stage = state.tasks[task_id].curr_stage;
    int spine_id = storage[std::make_tuple(task_id, stage, rank_from, rank_to)];
    int leaf_from = state.tasks[task_id].host_to_leaf[rank_from];
    state.input_queues[leaf_from][spine_id].push(data);
  }

  Ecmp(State &state) {
    int spine_ptr = 0;
    int spines = state.input_queues[0].size();
    for (int task_id = 0; task_id < state.tasks.size(); ++task_id) {
      auto &task = state.tasks[task_id];

      for (int stage = 0; stage < task.stages; ++stage) {
        int two_power = pow(2, stage);
        for (int host_from = 0; host_from < task.hosts; ++host_from) {
          int host_to = host_from ^ two_power;
          if (task.host_to_leaf[host_from] == task.host_to_leaf[host_to]) {
            continue;
          }
          storage[std::make_tuple(task_id, stage, host_from, host_to)] =
              spine_ptr;
          spine_ptr += 1;
          spine_ptr %= spines;
        }
      }
    }
  }
};