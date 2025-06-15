#include "algorithm"
#include "fort.hpp"
#include "random"
#include "set"
#include "single_include/nlohmann/json.hpp"
#include "thread_pool.hpp"
#include "unordered_map"
#include "vector"
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <graaflib/algorithm/coloring/greedy_graph_coloring.h>
#include <graaflib/algorithm/coloring/welsh_powell.h>
#include <graaflib/graph.h>
#include <graaflib/io/dot.h>
#include <iostream>
#include <limits.h>
#include <map>
#include <mutex>
#include <ranges>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>

const int seed = 42;
std::mt19937 rng(seed);

using namespace std;
using namespace std::filesystem;
using json = nlohmann::json;

string to_string(set<int> &s) {
  string res = "";
  for (int num : s) {
    res += format("{} ", num);
  }
  return res;
}

void write_ranking(vector<set<int>> &ranking) {
  fort::char_table table;
  table << fort::header;
  vector<string> header = {"Leaves", "Task"};
  table.range_write_ln(header.begin(), header.end());
  for (int leaf = 0; leaf < ranking.size(); ++leaf) {
    vector<string> body = {format("Leaf {}", leaf), to_string(ranking[leaf])};
    table.range_write_ln(body.begin(), body.end());
  }
  std::cout << table.to_string() << std::endl;
}

void write_task_ranking(vector<vector<set<int>>> &task_ranking) {
  fort::char_table table;
  table << fort::header;
  vector<string> header = {"Leaves"};
  for (int task = 0; task < task_ranking.size(); ++task) {
    header.push_back(format("Task {}", task));
  }
  table.range_write_ln(header.begin(), header.end());
  for (int leaf = 0; leaf < task_ranking[0].size(); ++leaf) {
    vector<string> body = {format("Leaf {}", leaf)};
    for (auto &task : task_ranking) {
      body.push_back(to_string(task[leaf]));
    }
    table.range_write_ln(body.begin(), body.end());
  }
  std::cout << table.to_string() << std::endl;
}

void print(vector<set<int>> &ranking) {

  for (int i = 0; i < ranking.size(); ++i) {
    cout << "leaf " << i << ": ";
    for (int host_rank : ranking[i]) {
      cout << host_rank << ' ';
    }
    cout << '\n';
  }
}

vector<set<int>> get_initial_ranking(int spines, int leaves, int hosts,
                                     double alpha = 1) {
  vector<set<int>> start_ranking(leaves);
  vector<double> probabilities(leaves, 1);
  //    std::mt19937 rng(seed);
  for (int host = 0; host < hosts; ++host) {
    discrete_distribution<> distribution(probabilities.begin(),
                                         probabilities.end());
    int host_leaf = distribution(rng);
    start_ranking[host_leaf].insert(host);
    if (spines > (int)start_ranking[host_leaf].size()) {
      probabilities[host_leaf] =
          pow(M_E, alpha * (double)start_ranking[host_leaf].size());
    } else {
      probabilities[host_leaf] = 0;
    }
  }
  return start_ranking;
}

double count_energy_1(vector<set<int>> &ranking, int spines, int leaves,
                      int hosts) {
  double res_energy = 0;
  int stages = (int)log2(hosts);
  vector<vector<int>> max_E(leaves);
  for (int leaf = 0; leaf < leaves; ++leaf) {
    max_E[leaf].resize(leaves - leaf, -1);
  }
  //    vector<vector<vector<int>>> E(stages);

  for (int stage = 0; stage < stages; ++stage) {
    //        E[stage].resize(leaves);
    for (int leaf = 0; leaf < leaves; ++leaf) {
      //            max_E[leaf].resize(leaves - leaf, -1);
      for (int next_leaf = leaf; next_leaf < leaves; ++next_leaf) {
        int counter = 0;
        for (auto host : ranking[leaf]) {
          int two_degree = pow(2, stage);
          if (ranking[next_leaf].contains(host ^ two_degree)) {
            counter += 1;
          }
        }
        max_E[leaf][next_leaf - leaf] =
            max(max_E[leaf][next_leaf - leaf], counter);
      }
    }
  }

  for (int leaf = 0; leaf < leaves; ++leaf) {
    if (ranking[leaf].empty())
      continue;
    double max_degree = 0;
    for (int other_leaf = 0; other_leaf < leaves; ++other_leaf) {
      if (other_leaf == leaf) {
        continue;
      }
      int leaf_less = leaf, leaf_more = other_leaf;
      if (leaf_less > leaf_more)
        swap(leaf_less, leaf_more);
      max_degree += max_E[leaf_less][leaf_more - leaf_less];
    }
    res_energy += (1 / (double)ranking[leaf].size()) *
                  max((double)0, max_degree - (double)ranking[leaf].size());
  }
  return res_energy;
}

vector<set<int>> get_swap_neighbor(vector<set<int>> &ranking, int task_id) {
  vector<int> not_empty_leaves_ids;
  for (int leaf = 0; leaf < ranking.size(); ++leaf) {
    if (ranking[leaf].empty())
      continue;
    not_empty_leaves_ids.push_back(leaf);
  }

  //    std::mt19937 rng(seed);
  int fill = (int)not_empty_leaves_ids.size();
  uniform_int_distribution<int> dist(0, fill - 1);
  int l1 = dist(rng), l2 = dist(rng);
  while (l1 == l2)
    l1 = dist(rng), l2 = dist(rng);
  l1 = not_empty_leaves_ids[l1];
  l2 = not_empty_leaves_ids[l2];
  uniform_int_distribution<int> dist_1(0, (int)ranking[l1].size() - 1);
  uniform_int_distribution<int> dist_2(0, (int)ranking[l2].size() - 1);
  auto it_host_1 = ranking[l1].begin();
  auto it_host_2 = ranking[l2].begin();
  advance(it_host_1, dist_1(rng));
  advance(it_host_2, dist_2(rng));

  int h1 = *it_host_1;
  int h2 = *it_host_2;
  // cout << format("swap {} and {} in task {}\n", h1, h2, task_id);
  // cout << "swap " << h1 << " and " << h2 << "in " <<'\n';
  auto new_ranking = ranking;
  new_ranking[l1].erase(h1);
  new_ranking[l2].erase(h2);
  new_ranking[l1].insert(h2);
  new_ranking[l2].insert(h1);
  return new_ranking;
}

bool task_is_congestionless_1(int spines, int leaves, int hosts, int task_id,
                              double (*schedule_temp)(int),
                              vector<set<int>> &ranking, int max_iter = 100) {
  //    std::mt19937 rng(seed);
  // auto ranking = get_initial_ranking(spines, leaves, hosts, alpha);
  // write_ranking(ranking);
  double prev_energy = count_energy_1(ranking, spines, leaves, hosts);
  uniform_real_distribution<double> distribution(0.0, 1.0);
  for (int iter = 0; iter < max_iter; ++iter) {
    // cout << '\n';
    // cout << "iter " << iter << '\n';
    // write_ranking(ranking);
    // cout << "energy = " << prev_energy << '\n';
    if (prev_energy == 0) {
      break;
    }
    double T = schedule_temp(iter);
    auto new_ranking = get_swap_neighbor(ranking, task_id);
    double new_energy = count_energy_1(new_ranking, spines, leaves, hosts);
    if (new_energy < prev_energy) {
      ranking = std::move(new_ranking);
      prev_energy = new_energy;
      continue;
    }
    if (pow(M_E, (prev_energy - new_energy) / T) > distribution(rng)) {
      ranking = std::move(new_ranking);
      prev_energy = new_energy;
    }
  }
  //    tuple<vector<set<int>>, bool>(ranking, prev_energy == 0);
  // return tuple<vector<set<int>>, double>(ranking, prev_energy);
  return prev_energy == 0;
}

int test1(int spines, int leaves, int tasks, double (*schedule_temp)(int),
          vector<vector<set<int>>> &ranking, int max_iter = 100) {
  // write_task_ranking(ranking);
  for (int task_id = 0; task_id < tasks; ++task_id) {
    int hosts = 0;
    for (auto &vec : ranking[task_id]) {
      hosts += vec.size();
    }
    if (!task_is_congestionless_1(spines, leaves, hosts, task_id, schedule_temp,
                                  ranking[task_id], max_iter)) {
      return 0;
    }
  }
  return 1;
}
double count_energy_overall(vector<vector<set<int>>> &ranking, int spines,
                            int leaves, int tasks, vector<int> &hosts_on_task) {
  double res_energy = 0;
  // int stages = (int)log2(hosts);
  vector<vector<vector<int>>> max_E(tasks, vector<vector<int>>(leaves));
  for (int task_id = 0; task_id < tasks; ++task_id) {
    for (int leaf = 0; leaf < leaves; ++leaf) {
      max_E[task_id][leaf].resize(leaves - leaf, -1);
    }
  }
  //    vector<vector<vector<int>>> E(stages);
  for (int task_id = 0; task_id < tasks; ++task_id) {
    int stages = (int)log2(hosts_on_task[task_id]);
    for (int stage = 0; stage < stages; ++stage) {
      //        E[stage].resize(leaves);
      int two_degree = pow(2, stage);
      for (int leaf = 0; leaf < leaves; ++leaf) {
        //            max_E[leaf].resize(leaves - leaf, -1);
        for (int next_leaf = leaf; next_leaf < leaves; ++next_leaf) {
          int counter = 0;
          for (auto host : ranking[task_id][leaf]) {
            if (ranking[task_id][next_leaf].contains(host ^ two_degree)) {
              counter += 1;
            }
          }
          max_E[task_id][leaf][next_leaf - leaf] =
              max(max_E[task_id][leaf][next_leaf - leaf], counter);
        }
      }
    }
  }

  // start energy counting
  for (int leaf = 0; leaf < leaves; ++leaf) {
    double max_degree = 0;
    for (int task_id = 0; task_id < tasks; ++task_id) {
      for (int other_leaf = 0; other_leaf < leaves; ++other_leaf) {
        if (other_leaf == leaf) {
          continue;
        }
        int leaf_less = leaf, leaf_more = other_leaf;
        if (leaf_less > leaf_more) {
          swap(leaf_less, leaf_more);
        }
        max_degree += max_E[task_id][leaf_less][leaf_more - leaf_less];
      }
    }
    res_energy += max((double)0, (double)max_degree - (double)spines);
  }
  return res_energy;
}
int test1_overall_annealling(int spines, int leaves, int tasks,
                             double (*schedule_temp)(int),
                             vector<vector<set<int>>> &ranking,
                             int max_iter = 100) {
  vector<int> hosts_on_task(tasks);
  for (int task_id = 0, hosts = 0; task_id < tasks; task_id++) {
    for (auto &leaf : ranking[task_id]) {
      hosts += leaf.size();
    }
    hosts_on_task[task_id] = hosts;
  }
  double prev_energy =
      count_energy_overall(ranking, spines, leaves, tasks, hosts_on_task);
  uniform_real_distribution<double> distribution(0.0, 1.0);
  uniform_int_distribution<int> uniform_distribution(0, tasks - 1);
  for (int iter = 0; iter < max_iter; ++iter) {
    if (prev_energy == 0) {
      break;
    }
    double T = schedule_temp(iter);
    int random_task_id = uniform_distribution(rng);
    auto new_task_ranking =
        get_swap_neighbor(ranking[random_task_id], random_task_id);
    auto new_ranking = ranking;
    new_ranking[random_task_id] = new_task_ranking;
    double new_energy =
        count_energy_overall(new_ranking, spines, leaves, tasks, hosts_on_task);
    if (new_energy < prev_energy) {
      ranking = std::move(new_ranking);
      prev_energy = new_energy;
      continue;
    }
    if (pow(M_E, (prev_energy - new_energy) / T) > distribution(rng)) {
      ranking = std::move(new_ranking);
      prev_energy = new_energy;
    }
  }
  return prev_energy == 0;
}

int split_num(int N, int k) {
  static map<pair<int, int>, int> results;
  if (k == 1)
    return 1;
  if (k == 2)
    return N - 1;
  if (results.contains(make_pair(N, k))) {
    return results[make_pair(N, k)];
  }

  int result = 0;
  for (int i = 1; i <= N - (k - 1); ++i) {
    result += split_num(N - i, k - 1);
  }
  results[make_pair(N, k)] = result;
  return result;
}

vector<int> random_split_n_on_k_summands(int N, int k) {
  vector<int> res;
  int K = k;
  int N_ = N;
  for (int i = 0; i < K - 1; ++i) {
    vector<int> probabilities;
    for (int num = 1; num < M_E * N / k; ++num) {
      probabilities.push_back(split_num(N - num, k - 1));
    }
    discrete_distribution<> distribution(probabilities.begin(),
                                         probabilities.end());
    int num = distribution(rng) + 1;
    res.push_back(num);
    N -= num;
    k--;
  }
  res.push_back(N);
  return res;
}

vector<vector<set<int>>> get_network_assignment(int spines, int leaves,
                                                int max_two_power_on_task,
                                                int tasks) {
  vector<vector<set<int>>> start_task_rankings(tasks);
  for (int task = 0; task < tasks; ++task) {
    uniform_int_distribution<int> dist(2, max_two_power_on_task);
    int random_two_power = pow(2, dist(rng));
    auto task_ranking =
        get_initial_ranking(spines, leaves, random_two_power, -1);
    //        auto it = max_element(task_ranking.begin(), task_ranking.end(),
    //                              [](set<int> &s1,
    //                              set<int> &s2) { return s1.size() <
    //                              s2.size(); });
    //        spines -= (int) (*it).size();
    start_task_rankings[task] = task_ranking;
  }
  return start_task_rankings;
}

auto prepare_stage_edges(vector<set<int>> &task_ranking,
                         vector<int> &host_to_leaf, int stage) {
  int two_degree = pow(2, stage);
  unordered_map<size_t, vector<tuple<int, int, int, int>>> stage_edges;
  unordered_map<size_t, int> leaves_of_size;
  for (int leaf_id = 0; leaf_id < task_ranking.size(); ++leaf_id) {
    auto &leaf = task_ranking[leaf_id];
    leaves_of_size[leaf.size()]++;
    for (int from_rank : leaf) {
      int to_rank = from_rank ^ two_degree;
      int to_leaf_id = host_to_leaf[to_rank];
      auto &to_leaf = task_ranking[to_leaf_id];
      stage_edges[min(leaf.size(), to_leaf.size())].emplace_back(
          leaf_id, to_leaf_id, from_rank, to_rank);
    }
  }
  return make_tuple(stage_edges, leaves_of_size);
}

bool try_kuhn(vector<map<int, vector<pair<int, int>>>> &Graph,
              vector<int> &matching, vector<bool> &used, int leaf_id) {
  if (used[leaf_id])
    return false;
  used[leaf_id] = true;
  for (auto &[to_leaf_id, _] : Graph[leaf_id]) {
    if (matching[to_leaf_id] == -1 ||
        try_kuhn(Graph, matching, used, matching[to_leaf_id])) {
      matching[to_leaf_id] = leaf_id;
      return true;
    }
  }
  return false;
}

void remove_matching_from_graph(vector<map<int, vector<pair<int, int>>>> &Graph,
                                vector<int> &matching, int color,
                                map<tuple<int, int, int>, int> &task_coloring,
                                int stage) {
  for (int to_id = 0; to_id < matching.size(); ++to_id) {
    int from_id = matching[to_id];
    if (from_id == -1)
      continue;
    auto [rank_from, rank_to] = Graph[from_id][to_id].back();
    task_coloring[make_tuple(stage, rank_from, rank_to)] = color;

    Graph[from_id][to_id].pop_back();
    if (Graph[from_id][to_id].empty()) {
      Graph[from_id].erase(to_id);
    }
    // auto it = Graph[from_id].find(to_id);
    // Graph[from_id].erase(it);
    // Graph[from_id].erase(to_id);
  }
}

// double kuhn(vector<multiset<int>> &Graph, int need_cover, int kuhn_times) {
//   double add_energy = 0;
//   vector<bool> used(Graph.size(), false);
//   for (int _ = 0; _ < kuhn_times; ++_) {

//     vector<int> matching(Graph.size(), -1);
//     for (int i = 0; i < Graph.size(); ++i) {
//       used.assign(Graph.size(), false);
//       try_kuhn(Graph, matching, used, i);
//     }

//     int matching_size = 0;
//     for (int from : matching) {
//       if (from != -1)
//         matching_size++;
//     }
//     add_energy += need_cover - matching_size;

//     remove_matching_from_graph(Graph, matching);
//   }
//   return add_energy;
// }

// double count_stage_energy(vector<set<int>> &task_ranking,
//                           vector<int> &host_to_leaf, int stage) {
//   vector<multiset<int>> Graph(task_ranking.size(), multiset<int>());
//   double stage_energy = 0;
//   auto prepare_stage = prepare_stage_edges(task_ranking, host_to_leaf,
//   stage); auto &stage_edges = get<0>(prepare_stage); auto &leaves_of_size =
//   get<1>(prepare_stage); int need_cover = 0; auto kv =
//   std::views::keys(stage_edges); vector<int> phases{kv.begin(), kv.end()};
//   sort(phases.begin(), phases.end());
//   for (int phase_id = (int)phases.size() - 1; phase_id >= 0; --phase_id) {
//     int phase_size = phases[phase_id];
//     auto &edges = stage_edges[phase_size];
//     need_cover += leaves_of_size[phase_size];
//     for (auto [from, to] : edges) {
//       Graph[from].insert(to);
//     }
//     int kuhn_times =
//         phase_id > 0 ? phase_size - phases[phase_id - 1] : phase_size;
//     stage_energy += kuhn(Graph, need_cover, kuhn_times);
//   }
//   return stage_energy;
// }

void kuhn_use_tor_switch(vector<map<int, vector<pair<int, int>>>> &Graph,
                         int kuhn_times, int phase_size,
                         map<tuple<int, int, int>, int> &task_coloring,
                         int stage) {
  // double add_energy = 0;
  vector<bool> used(Graph.size(), false);
  for (int color_id = 0; color_id < kuhn_times; ++color_id) {
    int color = phase_size - color_id;

    vector<int> matching(Graph.size(), -1);
    for (int i = 0; i < Graph.size(); ++i) {
      used.assign(Graph.size(), false);
      try_kuhn(Graph, matching, used, i);
    }

    // int matching_size = 0;
    // for (int from : matching) {
    //   if (from != -1)
    //     matching_size++;
    // }
    // add_energy += need_cover - matching_size;

    remove_matching_from_graph(Graph, matching, color, task_coloring, stage);
  }
  // return add_energy;
}

double count_stage_energy(vector<set<int>> &task_ranking,
                          vector<int> &host_to_leaf, int stage,
                          map<tuple<int, int, int>, int> &task_coloring,
                          bool interleaf_communication = true) {
  vector<map<int, vector<pair<int, int>>>> Graph(task_ranking.size());

  double stage_energy = 0;
  auto [stage_edges, leaves_of_size] =
      prepare_stage_edges(task_ranking, host_to_leaf, stage);
  // int need_cover = 0;
  auto kv = std::views::keys(stage_edges);
  vector<int> phases{kv.begin(), kv.end()};
  sort(phases.begin(), phases.end());
  for (int phase_id = (int)phases.size() - 1; phase_id >= 0; --phase_id) {
    int phase_size = phases[phase_id];
    auto &edges = stage_edges[phase_size];
    // need_cover += leaves_of_size[phase_size];
    for (auto [leaf_from, leaf_to, rank_from, rank_to] : edges) {

      if ((leaf_from != leaf_to) || (!interleaf_communication)) {
        // Graph[from].insert(to);
        Graph[leaf_from][leaf_to].emplace_back(rank_from, rank_to);
      }
    }
    int kuhn_times =
        phase_id > 0 ? phase_size - phases[phase_id - 1] : phase_size;
    kuhn_use_tor_switch(Graph, kuhn_times, phase_size, task_coloring, stage);
  }

  int edges_number = 0;
  for (auto &node : Graph) {
    edges_number += node.size();
  }
  // assert(stage_energy == edges_number);
  return edges_number;
}

auto count_energy(
    vector<set<int>> &task_ranking,
    double (*count_stage_energy_func)(vector<set<int>> &, vector<int> &, int,
                                      map<tuple<int, int, int>, int> &, bool)) {
  double energy = 0;
  map<tuple<int, int, int>, int> task_coloring;

  vector<int> host_to_leaf(1);
  //    set<int> leaf_sizes;
  int task_hosts = 0;
  for (int i = 0; i < task_ranking.size(); ++i) {
    auto &leaf = task_ranking[i];
    if (leaf.empty())
      continue;
    //        leaf_sizes.insert((int) leaf.size());
    task_hosts += (int)leaf.size();
    for (int rank : leaf) {
      if (rank >= host_to_leaf.size())
        host_to_leaf.resize(2 * rank);
      host_to_leaf[rank] = i;
    }
  }
  host_to_leaf.resize(task_hosts);
  int stages = (int)log2(task_hosts);
  if ((double)(stages) != log2(task_hosts)) {
    throw std::runtime_error("number of hosts on task is not a power of two");
  }
  for (int stage = 0; stage < stages; ++stage) {
    energy += count_stage_energy_func(task_ranking, host_to_leaf, stage,
                                      task_coloring, true);
  }
  return make_tuple(energy, task_coloring);
}

auto check_congestionless_ranking(vector<set<int>> &task_ranking, int task_id,
                                  double (*schedule_temp)(int),
                                  int max_iter = 100) {

  auto [prev_energy, prev_coloring] =
      count_energy(task_ranking, count_stage_energy);

  uniform_real_distribution<double> distribution(0.0, 1.0);
  for (int iter = 0; iter < max_iter; ++iter) {
    if (prev_energy == 0)
      break;
    double T = schedule_temp(iter);
    auto new_ranking = get_swap_neighbor(task_ranking, task_id);
    auto [new_energy, new_coloring] =
        count_energy(new_ranking, count_stage_energy);
    if (new_energy < prev_energy) {
      task_ranking = std::move(new_ranking);
      prev_energy = new_energy;
      prev_coloring = std::move(new_coloring);
      continue;
    }
    if (pow(M_E, (prev_energy - new_energy) / T) > distribution(rng)) {
      task_ranking = std::move(new_ranking);
      prev_energy = new_energy;
      prev_coloring = std::move(new_coloring);
    }
  }
  return make_tuple(prev_energy == 0, prev_coloring);
}

void add_clique(vector<pair<int, int>> &clique_nodes, int from,
                graaf::undirected_graph<pair<int, int>, int> &line_graph,
                map<pair<int, int>, size_t> &node_enumeration) {
  for (int i = from; i < clique_nodes.size(); ++i) {
    for (int j = i + 1; j < clique_nodes.size(); ++j) {
      auto node_from = clique_nodes[i];
      auto node_to = clique_nodes[j];
      //            auto [node_from_task, node_from_host] = clique_nodes[i];
      //            auto [node_to_task, node_to_host] = clique_nodes[j];
      line_graph.add_edge(node_enumeration[node_from],
                          node_enumeration[node_to], 1);
    }
  }
}

bool hypergraph_edge_coloring_exists(vector<vector<set<int>>> &ranking,
                                     int spines) {
  graaf::undirected_graph<pair<int, int>, int> line_graph;
  map<pair<int, int>, size_t> node_enumeration;

  for (int task_id = 0; task_id < ranking.size(); ++task_id) {
    auto &task = ranking[task_id];

    size_t max_hosts = 0;
    for (auto &task_leaf : task) {
      max_hosts = max(max_hosts, task_leaf.size());
    }
    for (int h = 1; h <= max_hosts; ++h) {
      auto p = make_pair(task_id, h);
      node_enumeration[p] = line_graph.add_vertex(p);
    }
  }

  vector<vector<pair<int, int>>> leaves_host_number_on_tasks(ranking[0].size());

  for (int task_id = 0; task_id < ranking.size(); ++task_id) {
    auto &task = ranking[task_id];
    for (int j = 0; j < task.size(); ++j) {
      auto &task_leaf = task[j];
      if (task_leaf.empty())
        continue;
      leaves_host_number_on_tasks[j].emplace_back(task_id, task_leaf.size());
    }
  }
  for (auto &leaf : leaves_host_number_on_tasks) {
    // sort(leaf.begin(), leaf.end(),
    //      [](const pair<int, int> &p1, const pair<int, int> &p2) {
    //        return p1.second < p2.second;
    //      });
    // for (int from = (int)leaf.size() - 1; from >= 0; --from) {
    //   if (from == 0 || leaf[from - 1].second < leaf[from].second) {
    //     add_clique(leaf, from, line_graph, node_enumeration);
    //   }
    // }
    for (auto [task_id, task_leaf_host_num] : leaf) {
      for (int i = 1; i <= task_leaf_host_num; ++i) {
        for (int j = i + 1; j <= task_leaf_host_num; ++j) {
          pair<int, int> node_1 = make_pair(task_id, i);
          pair<int, int> node_2 = make_pair(task_id, j);
          line_graph.add_edge(node_enumeration[node_1],
                              node_enumeration[node_2], 1);
        }
      }
    }
    for (int from = 0; from < leaf.size(); ++from) {
      for (int to = from + 1; to < leaf.size(); ++to) {
        auto [task_from, task_from_hosts] = leaf[from];
        auto [task_to, task_to_hosts] = leaf[to];
        for (int h1 = 1; h1 <= task_from_hosts; ++h1) {
          for (int h2 = 1; h2 <= task_to_hosts; ++h2) {
            pair<int, int> node_1 = make_pair(task_from, h1);
            pair<int, int> node_2 = make_pair(task_to, h2);
            line_graph.add_edge(node_enumeration[node_1],
                                node_enumeration[node_2], 1);
          }
        }
      }
    }
  }
  auto coloring = graaf::algorithm::greedy_graph_coloring(line_graph);
  auto kv = views::values(coloring);
  set<int> s(kv.begin(), kv.end());
  return s.size() <= spines;
}

auto test2(int spines, int leaves, int tasks, double (*schedule_temp)(int),
           vector<vector<set<int>>> &ranking, int max_iter = 100) {
  // auto ranking =
  //     get_network_assignment(spines, leaves, max_two_power_on_task, tasks);
  // write_task_ranking(ranking);
  // if (!hypergraph_edge_coloring_exists(ranking, spines)) {
  //   // cout << "can`t find proper coloring\n";
  //   return 2;
  // }
  map<tuple<int, int, int, int>, int> coloring;
  int result = 1;
  for (int task_id = 0; task_id < tasks; ++task_id) {
    auto [task_res, task_coloring] = check_congestionless_ranking(
        ranking[task_id], task_id, schedule_temp, max_iter);
    result *= task_res;
    for (auto &[edge, color] : task_coloring) {
      auto [stage, rank_from, rank_to] = edge;
      coloring[make_tuple(task_id, stage, rank_from, rank_to)] = color;
    }
  }
  return make_tuple(result, coloring);
  // write_task_ranking(ranking);
  // return 1;
}

vector<vector<set<int>>> get_ranking_from_json(int tasks, int leaves,
                                               auto &test) {
  vector<vector<set<int>>> res_ranking(tasks, vector<set<int>>(leaves));
  vector<int> tasks_cnt(tasks, 0);

  for (int task_id = 0; task_id < test.size(); ++task_id) {
    auto &task = test[task_id];

    for (int leaf_id = 0; leaf_id < task.size(); ++leaf_id) {
      int task_leaf_num = task[leaf_id];
      vector<int> V(task_leaf_num);
      iota(V.begin(), V.end(), tasks_cnt[task_id]);
      tasks_cnt[task_id] += task_leaf_num;
      res_ranking[task_id][leaf_id].insert(V.begin(), V.end());
    }
  }
  return res_ranking;
}
auto read_test(int tasks, int leaves, int spines, string alpha) {
  vector<vector<vector<set<int>>>> test_rankings;
  string test_name_dir = format("{}_{}_{}", tasks, leaves, spines);
  path test_file = current_path().parent_path();
  test_file.append("new_tests");
  test_file.append(test_name_dir);
  test_file.append(alpha);

  for (auto test_name : filesystem::directory_iterator{test_file}) {
    ifstream f(test_name.path());
    // cout << test_name.path() << '\n';
    json data = json::parse(f);
    for (auto &test : data) {
      test_rankings.push_back(get_ranking_from_json(tasks, leaves, test));
    }
    f.close();
  }

  return test_rankings;
}

auto run_tests_1(int tasks, int leaves, int spines, int max_iter = 10000,
                 int thread_num = 1) {
  double (*func)(int) = [](int iter) { return 10000 / double(iter + 1); };
  map<string, map<string, int>> statistic;
  string test_name_dir = format("{}_{}_{}", tasks, leaves, spines);
  path cwd = current_path().parent_path();
  cwd.append("tests");
  cwd.append(test_name_dir);

  BS::thread_pool pool(thread_num);
  mutex m;
  for (auto alpha_dir : filesystem::directory_iterator(cwd)) {
    string alpha = alpha_dir.path().filename().string();
    statistic[alpha] = {{"passed_tests", 0}, {"failed_tests", 0}};
    auto tests = read_test(tasks, leaves, spines, alpha);
    for (auto ranking : tests) {
      pool.submit(
          [&](auto test_tanking, string test_alpha) {
            auto res =
                test1(spines, leaves, tasks, func, test_tanking, max_iter);
            m.lock();
            switch (res) {
            case 0:
              statistic[test_alpha]["failed_tests"] += 1;
              break;
            case 1:
              statistic[test_alpha]["passed_tests"] += 1;
              break;
            }
            m.unlock();
          },
          ranking, alpha);
      // switch (test1(spines, leaves, tasks, func, ranking, max_iter)) {
      // case 0:
      //   statistic[alpha]["failed_tests"] += 1;
      //   break;
      // case 1:
      //   statistic[alpha]["passed_tests"] += 1;
      //   break;
      // }
    }
  }
  pool.wait_for_tasks();
  return statistic;
}

auto run_tests_2(int tasks, int leaves, int spines, int max_iter = 10000,
                 int thread_num = 1) {

  double (*func)(int) = [](int iter) { return 10000 / double(iter + 1); };

  map<string, map<string, int>> statistic;
  string test_name_dir = format("{}_{}_{}", tasks, leaves, spines);
  path cwd = current_path().parent_path();
  cwd.append("tests");
  cwd.append(test_name_dir);

  BS::thread_pool pool(thread_num);
  mutex m;
  for (auto alpha_dir : filesystem::directory_iterator(cwd)) {
    string alpha = alpha_dir.path().filename().string();
    statistic[alpha] = {
        {"passed_tests", 0}, {"failed_tests", 0}, {"not_found_coloring", 0}};

    auto tests = read_test(tasks, leaves, spines, alpha);
    for (auto ranking : tests) {
      pool.submit(
          [&](auto test_ranking, string test_alpha) {
            auto [res, _] =
                test2(spines, leaves, tasks, func, test_ranking, max_iter);
            m.lock();
            switch (res) {
            case 0:
              statistic[test_alpha]["failed_tests"] += 1;
              break;
            case 1:
              statistic[test_alpha]["passed_tests"] += 1;
              break;
            case 2:
              statistic[test_alpha]["not_found_coloring"] += 1;
              statistic[test_alpha]["failed_tests"] += 1;
              break;
            }
            m.unlock();
          },
          ranking, alpha);
    }
  }
  pool.wait_for_tasks();
  return statistic;
}

template <ranges::range R> constexpr auto to_vector(R &&r) {
  using elem_t = std::decay_t<ranges::range_value_t<R>>;
  return std::vector<elem_t>{r.begin(), r.end()};
}
void run_all_tests(string stats_filename, auto &resolve_func, int thread_num) {
  map<string, map<string, map<string, int>>> all_stats;
  path tests_dir = current_path().parent_path();
  tests_dir.append("new_tests");
  int tasks = 0, leaves = 0, spines = 0;
  for (auto tls : filesystem::directory_iterator(tests_dir)) {
    auto parts = tls.path().filename().string() | views::split('_') |
                 std::views::transform(
                     [](auto r) { return std::string(r.data(), r.size()); });
    auto asVector = std::vector(parts.begin(), parts.end());
    tasks = stoi(asVector[0]);
    leaves = stoi(asVector[1]);
    spines = stoi(asVector[2]);
    all_stats[format("({}, {}, {})", tasks, leaves, spines)] =
        resolve_func(tasks, leaves, spines, 10000, thread_num);
  }

  json stat_json(all_stats);
  path cwd = current_path().parent_path();
  cwd.append(stats_filename);
  ofstream file(cwd);
  file << stat_json;
  file.close();
}

auto debug_test2(int tasks, int leaves, int spines, string alpha) {
  auto tests = read_test(tasks, leaves, spines, alpha);
  tasks = 1;
  double (*func)(int) = [](int iter) { return 10000 / double(iter + 1); };
  BS::thread_pool pool(58);
  mutex m;

  // auto ranking = tests[3];
  vector<int> results(tests.size());
  for (int i = 0; i < tests.size(); ++i) {
    auto ranking = tests[i];
    pool.submit(
        [&](auto test_ranking, int pos) {
          auto [res, _] =
              test2(spines, leaves, tasks, func, test_ranking, 10000);
          m.lock();
          results[pos] = res;
          m.unlock();
        },
        ranking, i);
  }
  pool.wait_for_tasks();
  return results;
}

auto read_colored_test(int tasks, int leaves, int spines, path test_file) {
  vector<vector<vector<set<int>>>> test_rankings;
  vector<bool> test_colorings;
  // string test_name_dir = format("{}_{}_{}", tasks, leaves, spines);
  // path test_file = current_path().parent_path();
  // test_file.append(test_dir_name);
  // test_file.append(test_name_dir);
  // test_file.append(alpha);

  int count = std::distance(std::filesystem::directory_iterator(test_file),
                            std::filesystem::directory_iterator{});
  count /= 2;
  // cout << count;
  for (int num = 0; num < count; ++num) {
    auto current_test_file = test_file;
    current_test_file.append(format("test_{}.json", num));
    ifstream f(current_test_file);
    json data = json::parse(f);
    for (auto &test : data) {
      test_rankings.push_back(get_ranking_from_json(tasks, leaves, test));
    }
    f.close();

    auto current_color_file = test_file;
    current_color_file.append(format("color_{}.json", num));
    ifstream f2(current_color_file);
    json color_data = json::parse(f2);
    for (auto &color : color_data) {
      bool res_color = color.get<bool>();
      test_colorings.push_back(res_color);
    }
    f2.close();
  }

  return make_tuple(test_rankings, test_colorings);
}
auto colored_test2(int tasks, int leaves, int spines, string alpha,
                   string test_dir_name) {
  string test_name_dir = format("{}_{}_{}", tasks, leaves, spines);
  path test_file = current_path().parent_path();
  test_file.append(test_dir_name);
  test_file.append(test_name_dir);
  test_file.append(alpha);
  auto [tests, colorings] = read_colored_test(tasks, leaves, spines, test_file);
  double (*func)(int) = [](int iter) { return 10000 / double(iter + 1); };
  BS::thread_pool pool(40);
  mutex m;

  // auto ranking = tests[3];
  vector<int> results(tests.size());
  for (int i = 0; i < tests.size(); ++i) {
    if (!colorings[i]) {
      results[i] = 0;
      continue;
    }
    auto ranking = tests[i];
    pool.submit(
        [&](auto test_ranking, int pos) {
          auto [res, _] =
              test2(spines, leaves, tasks, func, test_ranking, 10000);
          m.lock();
          results[pos] = res;
          m.unlock();
        },
        ranking, i);
  }
  pool.wait_for_tasks();
  return results;
}

auto colored_test1(int tasks, int leaves, int spines, string alpha,
                   string test_dir_name) {
  string test_name_dir = format("{}_{}_{}", tasks, leaves, spines);
  path test_file = current_path()
                       .parent_path()
                       .append(test_dir_name)
                       .append(test_name_dir)
                       .append(alpha);

  auto [tests, colorings] = read_colored_test(tasks, leaves, spines, test_file);
  double (*func)(int) = [](int iter) { return 10000 / double(iter + 1); };
  BS::thread_pool pool(60);
  mutex m;

  // auto ranking = tests[3];
  vector<int> results(tests.size());
  for (int i = 0; i < tests.size(); ++i) {
    auto ranking = tests[i];
    pool.submit(
        [&](auto test_ranking, int pos) {
          auto res = test1(spines, leaves, tasks, func, test_ranking, 10000);
          m.lock();
          results[pos] = res;
          m.unlock();
        },
        ranking, i);
  }
  pool.wait_for_tasks();
  return results;
}

double count_stage_energy_cota_up_to_down(vector<set<int>> task_ranking,
                                          vector<int> &host_to_leaf,
                                          int stage) {
  sort(
      task_ranking.begin(), task_ranking.end(),
      [](const set<int> &l, const set<int> &r) { return l.size() < r.size(); });

  vector<multiset<int>> Graph(task_ranking.size(), multiset<int>());
  double stage_energy = 0;
  int two_degree = pow(2, stage);
  for (int leaf_id = 0; leaf_id < task_ranking.size(); ++leaf_id) {
    auto &leaf = task_ranking[leaf_id];
    // leaves_of_size[leaf.size()]++;
    for (int host_rank : leaf) {
      int to_rank = host_rank ^ two_degree;
      int to_leaf_id = host_to_leaf[to_rank];
      auto &to_leaf = task_ranking[to_leaf_id];
      // stage_edges[min(leaf.size(), to_leaf.size())].emplace_back(leaf_id,
      //                                                            to_leaf_id);
      Graph[leaf_id].insert(to_leaf_id);
    }
  }
  return -1;
}
void mfp_strategy(vector<map<int, int>> &Graph, int colors,
                  map<int, int> &node_degrees) {
  for (int color = 0; color < colors; ++color) {
    vector<pair<int, int>> degree_order;
    set<int> used_dest;
    for (auto [node, degree] : node_degrees) {
      degree_order.emplace_back(degree, node);
    }
    sort(degree_order.begin(), degree_order.end());
    for (auto [degree, node] : degree_order) {
      int node_to = -1;
      int flows_to = INT_MAX;

      for (auto [leaf_to, times] : Graph[node]) {
        if (used_dest.contains(leaf_to))
          continue;
        if (times < flows_to) {
          flows_to = times;
          node_to = leaf_to;
        }
      }
      if (-1 == node_to)
        continue;

      used_dest.insert(node_to);
      Graph[node][node_to]--;
      if (Graph[node][node_to] == 0) {
        Graph[node].erase(node_to);
      }
      node_degrees[node]--;
    }
  }
}
double count_stage_energy_cota_down_to_up(
    vector<set<int>> &task_ranking, vector<int> &host_to_leaf, int stage,
    map<tuple<int, int, int>, int> &task_coloring,
    bool interleaf_communication = true) {

  vector<map<int, int>> Graph(task_ranking.size(), map<int, int>());
  double stage_energy = 0;
  auto [stage_edges, leaves_of_size] =
      prepare_stage_edges(task_ranking, host_to_leaf, stage);
  auto kv = std::views::keys(stage_edges);
  vector<int> phases{kv.begin(), kv.end()};
  sort(phases.begin(), phases.end());

  map<int, int> node_degrees;

  for (int phase_id = (int)phases.size() - 1; phase_id >= 0; --phase_id) {
    int phase_size = phases[phase_id];
    auto &edges = stage_edges[phase_size];
    // need_cover += leaves_of_size[phase_size];
    for (auto &[from, to, rank_from, rank_to] : edges) {
      if ((from != to) || (!interleaf_communication)) {
        Graph[from][to]++;
        node_degrees[from]++;
      }
    }
    int new_colors =
        phase_id > 0 ? phase_size - phases[phase_id - 1] : phase_size;
    mfp_strategy(Graph, new_colors, node_degrees);
  }
  int edges_number = 0;
  for (auto [_, degree] : node_degrees) {
    edges_number += degree;
  }
  return edges_number;
}
auto read_test_from_file(int tasks, int leaves, int spines, path test_file) {
  vector<vector<vector<set<int>>>> test_rankings;
  // auto current_test_file = test_file;
  // current_test_file.append(format("test_{}.json", num));
  ifstream f(test_file);
  json data = json::parse(f);
  for (auto &test : data) {
    test_rankings.push_back(get_ranking_from_json(tasks, leaves, test));
  }
  f.close();
  return test_rankings;
}
int main() {
  // auto [tasks, leaves, spines] = make_tuple(1, 2, 8);
  // path test_file = current_path().parent_path().append("train_test.json");
  // auto tests = read_test_from_file(tasks, leaves, spines, test_file);
  // double (*func)(int) = [](int iter) { return 1000000 / double(iter + 1); };
  // for (auto test : tests) {
  //   cout << test1(spines, leaves, tasks, func, test, 1000000) << '\n';
  //   // cout << test1_overall_annealling(spines, leaves, tasks, func, test,
  //   // 1000000)
  //   //      << '\n';
  // }
  vector<string> alpha_list = {"-1.0", "-0.9", "-0.8", "-0.7", "-0.6", "-0.5",
                               "-0.4", "-0.3", "-0.2", "-0.1", "0.0",  "0.1",
                               "0.2",  "0.3",  "0.4",  "0.5",  "0.6",  "0.7",
                               "0.8",  "0.9",  "1.0"};

  double (*func)(int) = [](int iter) { return 1000000 / double(iter + 1); };
  vector<tuple<int, int, int>> t_l_s = {
      {1, 2, 16},  {1, 3, 16},  {1, 4, 16},  {1, 5, 16},  {1, 6, 16},
      {1, 7, 16},  {1, 8, 16},  {1, 9, 16},  {1, 10, 16}, {1, 11, 16},
      {1, 12, 16}, {1, 13, 16}, {1, 14, 16}, {1, 15, 16}, {1, 16, 16}};

  for (auto [tasks, leaves, spines] : t_l_s) {
    string t_l_s_dir = format("{}_{}_{}", tasks, leaves, spines);

    map<string, vector<int>> alpha_results;

    for (string alpha : alpha_list) {
      path test_file_dir = current_path()
                               .parent_path()
                               .append("new_single_task_tests")
                               .append(t_l_s_dir)
                               .append(alpha);

      int count =
          std::distance(std::filesystem::directory_iterator(test_file_dir),
                        std::filesystem::directory_iterator{});
      for (int file_num = 0; file_num < count; ++file_num) {
        path output_file = current_path()
                               .parent_path()
                               .append("new_single_task_tests_scenario1")
                               .append(t_l_s_dir)
                               .append(alpha)
                               .append(format("test_results{}.json", file_num));
        if (filesystem::exists(output_file)) {
          continue;
        }
        cout << output_file << endl;

        path test_file = current_path()
                             .parent_path()
                             .append("new_single_task_tests")
                             .append(t_l_s_dir)
                             .append(alpha)
                             .append(format("test_{}.json", file_num));
        auto tests = read_test_from_file(tasks, leaves, spines, test_file);

        BS::thread_pool pool(4);
        mutex m;
        vector<int> results(tests.size());
        for (int i = 0; i < tests.size(); ++i) {
          auto ranking = tests[i];
          pool.submit(
              [&](auto test_ranking, int pos) {
                auto res =
                    test1(spines, leaves, tasks, func, test_ranking, 1000000);
                m.lock();
                results[pos] = res;
                m.unlock();
              },
              ranking, i);
        }
        pool.wait_for_tasks();

        json results_json(results);
        output_file = current_path()
                          .parent_path()
                          .append("new_single_task_tests_scenario1")
                          .append(t_l_s_dir)
                          .append(alpha);
        filesystem::create_directories(output_file);
        output_file.append(format("test_results{}.json", file_num));

        ofstream results_file(output_file);
        results_file << results_json;
        results_file.close();
      }
    }
  }

  // vector<tuple<int, int, int>> t_l_s = {
  //     // {2, 8, 4},
  //     // {3, 8, 4},
  //     // {4, 8, 4},
  //     //
  //     {2, 16, 8},
  //     {3, 16, 8},
  //     // {4, 16, 8},
  //     // {5, 16, 8},
  //     // {6, 16, 8},
  //     {7, 16, 8},
  //     // {8, 16, 8},
  //     // //
  //     // {2, 32, 16},
  //     // {3, 32, 16},
  //     // {4, 32, 16},
  //     // {5, 32, 16},
  //     // {6, 32, 16},
  //     // {7, 32, 16},
  //     // {8, 32, 16},
  //     // {9, 32, 16},
  //     // {10, 32, 16},
  //     // {11, 32, 16},
  //     // {12, 32, 16},
  //     // {13, 32, 16},
  //     // {14, 32, 16},
  //     // {15, 32, 16},
  //     // {16, 32, 16}
  // };

  // double (*func)(int) = [](int iter) { return 1000000 / double(iter + 1); };
  // for (auto [tasks, leaves, spines] : t_l_s) {
  //   string test_name_dir = format("{}_{}_{}", tasks, leaves, spines);
  //   path test_file = current_path().parent_path();
  //   test_file.append("new_special_multitests").append(test_name_dir);
  //   int count = std::distance(std::filesystem::directory_iterator(test_file),
  //                             std::filesystem::directory_iterator{});
  //   for (int file_num = 0; file_num < count / 2; ++file_num) {
  //     path output_file = current_path()
  //                            .parent_path()
  //                            .append("new_special_multitests_scenario2")
  //                            .append(test_name_dir)
  //                            .append(format("test_{}.json", file_num));
  //     if (filesystem::exists(output_file)) {
  //       continue;
  //     }

  //     auto num_test_file = test_file;
  //     num_test_file.append(format("test_{}.json", file_num));
  //     auto tests_rankings =
  //         read_test_from_file(tasks, leaves, spines, num_test_file);

  //     vector<int> results(tests_rankings.size());
  //     vector<map<tuple<int, int, int, int>, int>> colorings(
  //         tests_rankings.size());
  //     vector<vector<vector<set<int>>>> final_rankings(tests_rankings.size());

  //     BS::thread_pool pool(4);
  //     mutex m;
  //     for (int i = 0; i < tests_rankings.size(); ++i) {
  //       auto ranking = tests_rankings[i];
  //       pool.submit(
  //           [&](auto test_ranking, int pos) {
  //             auto [result, coloring] =
  //                 test2(spines, leaves, tasks, func, test_ranking, 1000000);
  //             m.lock();
  //             results[pos] = result;
  //             colorings[pos] = coloring;
  //             final_rankings[pos] = test_ranking;
  //             m.unlock();
  //           },
  //           ranking, i);
  //     }
  //     pool.wait_for_tasks();

  //     json results_json(results);
  //     output_file = current_path()
  //                       .parent_path()
  //                       .append("new_special_multitests_scenario2")
  //                       .append(test_name_dir);

  //     filesystem::create_directories(output_file);

  //     output_file.append(format("test_{}.json", file_num));
  //     ofstream results_file(output_file);
  //     results_file << results_json;
  //     results_file.close();

  //     json colorings_json(colorings);
  //     path output_colorings = current_path()
  //                                 .parent_path()
  //                                 .append("new_special_multitests_scenario2")
  //                                 .append(test_name_dir);

  //     filesystem::create_directories(output_colorings);
  //     output_colorings.append(format("coloring_{}.json", file_num));
  //     ofstream colorings_file(output_colorings);
  //     colorings_file << colorings_json;
  //     colorings_file.close();

  //     json rankings_json(final_rankings);
  //     path output_rankings = current_path()
  //                                .parent_path()
  //                                .append("new_special_multitests_scenario2")
  //                                .append(test_name_dir);

  //     filesystem::create_directories(output_rankings);
  //     output_rankings.append(format("ranking_{}.json", file_num));
  //     ofstream rankings_file(output_rankings);
  //     rankings_file << rankings_json;
  //     rankings_file.close();
  //   }
  // }
}
