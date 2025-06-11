#include "../core.hpp"
#include <map>
struct Fixed : public Decider {
  std::map<std::tuple<int, int, int, int>, int> storage;

  void decide(Data data, State &state) override {
    auto [task_id, rank_from, rank_to] = data;
    int stage = state.tasks[task_id].curr_stage;
    int spine_id = storage[std::make_tuple(task_id, stage, rank_from, rank_to)];
    int leaf_from = state.tasks[task_id].host_to_leaf[rank_from];
    state.input_queues[leaf_from][spine_id].push(data);
  }
  Fixed(const std::map<std::tuple<int, int, int, int>, int> &storage)
      : storage(storage) {}
};