#include "../core.hpp"
#include <random>

struct Rand : public Decider {
  std::mt19937 rng;
  void decide(Data data, State &state) override {
    auto [task_id, rank_from, rank_to] = data;
    int leaf_from = state.tasks[task_id].host_to_leaf[rank_from];
    int spines = state.input_queues[0].size();

    std::uniform_int_distribution<int> dist(0, spines - 1);
    int spine_id = dist(rng);

    state.input_queues[leaf_from][spine_id].push(data);
  }

  Rand(int seed) { rng = std::mt19937(seed); }
};