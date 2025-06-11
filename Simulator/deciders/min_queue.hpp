#include "../core.hpp"

struct Min_queue : public Decider {

  void decide(Data data, State &state) override {
    auto [task_id, rank_from, rank_to] = data;
    int leaf_from = state.tasks[task_id].host_to_leaf[rank_from];
    int spines = state.input_queues[0].size();

    int min_size = INT_MAX;
    int min_spine_id = -1;
    for (int spine_id = 0; spine_id < spines; ++spine_id) {
      auto &spine = state.input_queues[leaf_from][spine_id];
      if (spine.size() < min_size) {
        min_size = spine.size();
        min_spine_id = spine_id;
      }
    }
    // simulator->spine_queues[min_spine_id].push(data);
    state.input_queues[leaf_from][min_spine_id].push(data);
  }
};