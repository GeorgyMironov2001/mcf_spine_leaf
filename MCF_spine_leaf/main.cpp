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

auto read_colored_test(int tasks, int leaves, int spines, string alpha,
                       string test_dir_name = "multi_tasks_tests") {
  vector<vector<vector<set<int>>>> test_rankings;
  vector<bool> test_colorings;
  string test_name_dir = format("{}_{}_{}", tasks, leaves, spines);
  path test_file = current_path().parent_path();
  test_file.append(test_dir_name);
  test_file.append(test_name_dir);
  test_file.append(alpha);

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
  auto [tests, colorings] =
      read_colored_test(tasks, leaves, spines, alpha, test_dir_name);
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
  auto [tests, colorings] =
      read_colored_test(tasks, leaves, spines, alpha, test_dir_name);
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

int main() {

  vector<vector<set<int>>> test_ranking = {
      {{{46, 60,  95,  96,  65, 118, 6,  80, 47, 97, 106, 68, 125, 114, 73, 26,
         58, 117, 123, 100, 93, 63,  61, 17, 20, 49, 43,  9,  111, 91,  24, 62},
        {77, 21,  120, 82, 39, 18, 94, 67, 13, 55, 90, 5, 12,
         27, 121, 8,   75, 7,  41, 74, 50, 3,  29, 81, 78},
        {59, 99, 113, 11,  112, 31, 127, 30, 37, 42, 83, 28, 71, 105, 122,
         87, 64, 16,  101, 23,  57, 69,  40, 0,  85, 79, 14, 86, 70},
        {38,  1,   2,  33, 54, 124, 84, 15, 88, 51, 44,  72, 110, 104,
         102, 119, 45, 36, 56, 76,  48, 19, 32, 53, 115, 66, 25},
        {52, 10, 34, 35, 107, 4, 22, 98, 109, 92, 126, 108, 103, 116, 89},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}}},
      {{{},
        {106, 82, 68, 52, 40, 84, 35},
        {22, 64, 74},
        {20, 67, 15, 24, 44},
        {71, 87, 116, 114, 14, 21, 62, 12, 59, 112, 6, 41, 66, 34, 23, 125, 88},
        {78,  102, 69, 43, 81, 25, 79,  30,  77, 33, 93, 91, 13, 16,  28,  2,
         113, 118, 46, 86, 29, 72, 127, 121, 55, 73, 4,  18, 97, 103, 107, 51},
        {70, 104, 8,  53, 0,  37,  36,  39, 45, 99, 126, 26, 60, 83, 115, 122,
         19, 119, 56, 96, 85, 111, 108, 80, 57, 48, 50,  1,  89, 95, 42,  11},
        {49, 5,   17, 100, 27, 32,  120, 94, 7,   92, 123, 65, 10,
         61, 110, 3,  90,  76, 124, 9,   54, 101, 75, 31,  58, 117},
        {38, 47, 109, 98, 105, 63},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}}},
      {{{},
        {},
        {},
        {},
        {},
        {},
        {},
        {107, 37, 1, 95, 33, 8},
        {91, 84, 59,  19, 70,  61, 125, 102, 78, 123, 25,  14, 7,
         11, 50, 117, 52, 103, 30, 114, 54,  62, 4,   112, 81, 96},
        {110, 69, 76, 77, 80,  56, 73, 68, 89, 45, 71, 49, 24, 86,  43, 121,
         38,  57, 28, 79, 127, 67, 17, 41, 51, 87, 36, 18, 75, 109, 42, 15},
        {40,  47,  26, 124, 13,  94,  9,   31,  66,  88,  93,
         104, 97,  98, 64,  3,   35,  100, 120, 111, 108, 23,
         48,  101, 82, 2,   119, 106, 113, 16,  118, 122},
        {90, 27, 5, 83, 21, 85,  58, 63, 116, 65,
         46, 53, 0, 60, 6,  105, 39, 55, 74,  72},
        {44, 32, 20, 12, 29, 92},
        {34, 126, 99, 10, 115, 22},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}}},
      {{{},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {43, 89, 28, 109, 45, 40, 83, 80, 42, 0, 103, 82},
        {97, 1,  8,  32, 9,  58, 96, 114, 35, 24,  108, 59, 66,
         18, 78, 64, 67, 23, 46, 2,  117, 14, 101, 119, 73, 110},
        {30,  34, 115, 55, 122, 38, 48,  102, 39,  79, 53, 65,  86,
         121, 77, 76,  50, 4,   29, 104, 3,   100, 68, 21, 127, 107},
        {5,  93, 71, 63, 26,  99, 49, 120, 75, 94, 85, 15, 47, 51,
         31, 20, 12, 6,  123, 70, 92, 90,  37, 57, 52, 81, 105},
        {54, 106, 27, 13, 61, 126, 91, 87, 84, 44, 17, 69, 33, 22, 125, 25, 72},
        {88, 111, 19, 16, 98, 62, 41, 118, 112, 60, 74, 113, 95},
        {11, 10, 36, 124, 56, 7, 116},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}}},
      {{{},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {83, 85, 102, 29, 61},
        {14, 55, 95, 110, 92, 38, 26, 56, 89, 24, 122, 11, 88, 125, 23},
        {98, 97, 112, 8, 63, 40, 73, 105, 5, 78, 16, 96, 101, 67, 12, 75, 84,
         58, 28},
        {115, 33,  93,  87, 86, 103, 123, 30, 27, 4,   59, 22, 54,
         118, 111, 109, 50, 62, 108, 57,  41, 46, 124, 91, 36},
        {119, 0,  43,  9,   79, 48, 15, 39, 32, 100, 6,
         76,  25, 126, 107, 81, 82, 94, 17, 44, 99,  68},
        {116, 18,  80,  49, 2,  7,  31, 90,  106, 66, 3,   127, 69, 120,
         70,  117, 114, 34, 51, 19, 65, 121, 52,  71, 104, 10,  77},
        {37, 47, 53, 113, 1, 42, 35, 74, 21, 72, 13, 45, 20},
        {},
        {64, 60},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}}},
      {{{},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {125, 61, 75, 31, 42, 0, 112, 70, 88, 104},
        {50, 111, 65, 10, 102},
        {114, 67, 124, 8, 52, 110, 48, 54, 45, 90, 83, 58, 81, 3, 47, 46, 77,
         76, 20},
        {63, 92, 19, 12,  93, 72, 105, 116, 74, 39,  126, 41, 82, 33,  30, 56,
         4,  21, 53, 103, 32, 86, 109, 115, 71, 100, 27,  22, 98, 118, 24, 113},
        {34, 68, 5,  73,  89, 28, 120, 35, 94, 91, 43, 2,
         26, 96, 13, 107, 44, 78, 85,  80, 49, 97, 7},
        {51,  57,  99, 64, 29,  25, 66, 18, 40,  23,  16, 122,
         121, 108, 95, 62, 123, 6,  15, 79, 119, 101, 127},
        {84, 17, 87, 37, 9, 14, 69, 106, 36, 1, 11, 59, 38, 55, 60, 117},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}}},
      {{{},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {14, 103, 39, 107, 15, 90, 69},
        {126, 55, 100, 118, 23, 63, 22, 91, 84},
        {97, 3, 71, 68, 110, 32, 104, 49, 102, 25, 45, 54, 35, 52, 43, 46},
        {72, 101, 37, 29, 83, 34, 11, 2,   120, 56, 53, 119, 66, 42, 116, 9,
         20, 0,   5,  24, 75, 76, 62, 115, 127, 64, 10, 122, 99, 31, 36,  48},
        {74, 93, 113, 95, 33, 88,  67, 105, 112, 60, 7,  41, 27, 44, 38,
         47, 73, 82,  21, 86, 124, 6,  125, 77,  58, 65, 30, 80, 98},
        {13, 8, 87, 108, 4, 96, 78, 1, 117, 18, 111, 94, 51, 123, 70, 114, 59,
         92},
        {57, 50, 40, 16, 106, 79, 12, 17, 85, 89, 26, 81, 28, 109, 19, 121, 61},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}}},
      {{{},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {84, 30, 64},
        {110, 23, 115, 86, 18, 122, 13, 47, 20, 120, 7, 0, 73, 87},
        {70, 43, 99, 88, 63, 119, 65, 90, 97, 33, 51, 6, 44, 37, 5},
        {71,  25,  31,  112, 80,  103, 3,  8,  28, 82, 91,
         92,  32,  109, 40,  74,  102, 66, 4,  69, 45, 76,
         111, 104, 89,  100, 101, 41,  19, 39, 14, 75},
        {107, 126, 10, 61, 113, 17,  56, 124, 42,  106, 96, 1,  2, 34,
         78,  59,  50, 9,  98,  125, 81, 27,  118, 95,  94, 67, 38},
        {121, 68, 26, 36, 127, 12, 11, 105, 114, 29, 62,
         46,  60, 72, 35, 48,  24, 57, 108, 52,  79},
        {21, 93, 83, 117, 58, 77, 116, 49, 85, 22, 54, 15, 16, 53, 123, 55},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}}},
      {{{},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {27, 101, 56, 45, 36},
        {30, 87, 41, 4, 121, 109, 7, 24, 0, 75, 33},
        {106, 26, 107, 57, 47, 66, 50, 113, 127, 67, 114, 61, 88, 35, 29, 126},
        {1,  90,  49, 25, 120, 102, 37, 11, 97, 9,  96,  117, 98, 72, 93, 21,
         77, 110, 65, 62, 22,  69,  19, 83, 43, 17, 118, 5,   76, 79, 16, 60},
        {42, 82,  55, 92, 105, 12,  38,  116, 124, 8,  34,
         70, 123, 73, 20, 80,  119, 122, 3,   85,  6,  100,
         53, 74,  91, 63, 59,  71,  31,  115, 14,  104},
        {99, 10, 51, 46, 18, 78, 44, 95, 108, 32, 103, 112, 28,
         89, 48, 64, 39, 15, 2,  81, 23, 94,  68, 13,  86,  52},
        {84, 54, 40, 58, 125, 111},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}}},
      {{{},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {9, 13, 82, 46, 7, 114},
        {26, 45,  103, 18, 30, 104, 58,  17, 121, 48,  43, 110, 77,
         28, 118, 47,  15, 74, 4,   111, 91, 51,  108, 31, 116, 109},
        {117, 42,  113, 125, 34, 33,  38, 93,  105, 44, 106,
         63,  101, 94,  84,  87, 98,  35, 102, 1,   37, 120,
         67,  14,  3,   86,  23, 100, 8,  52,  36,  126},
        {19, 115, 62, 81, 50, 71, 57, 123, 83, 53, 78, 99, 92, 59, 96,  54,
         20, 119, 79, 76, 0,  95, 29, 122, 5,  49, 75, 65, 89, 40, 124, 107},
        {127, 80, 112, 61, 70, 12, 25, 66, 97, 11, 55, 32,
         10,  73, 22,  41, 88, 85, 72, 90, 2,  39, 68},
        {6, 64, 24, 60, 21},
        {27, 16, 69, 56},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}}},
      {{{},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {102, 22, 43, 37, 126, 2, 99, 83, 42},
        {111, 69, 29, 98, 65, 58,  23,  54, 71, 60, 82,  31, 0,  20,
         49,  36, 17, 18, 14, 117, 118, 77, 44, 1,  108, 93, 120},
        {106, 5,  52,  9,  21,  103, 63, 75, 35, 59, 124, 116, 46,  85,
         4,   30, 113, 88, 100, 112, 87, 34, 84, 64, 115, 79,  114, 41},
        {25, 15, 48, 7,  61, 66, 32, 53, 39, 81, 91, 28,  47, 96,
         24, 57, 51, 16, 38, 67, 97, 50, 76, 56, 94, 107, 27, 12},
        {127, 104, 3,  119, 40, 62, 121, 89,  10, 78, 6, 105, 109, 123,
         74,  80,  72, 92,  13, 95, 45,  101, 68, 73, 8, 90,  110},
        {26, 125, 19, 55, 122, 33, 11, 86, 70},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}}},
      {{{},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {56, 50, 95, 44},
        {123, 82, 109, 47, 117},
        {110, 118, 68, 98, 33, 80, 85, 2,  107, 119, 76, 126,
         77,  92,  35, 86, 46, 55, 93, 41, 100, 104, 122},
        {45,  91, 24, 73,  3,  19, 34, 52, 108, 6,  15, 99, 31, 65, 94, 17,
         105, 25, 38, 120, 10, 63, 70, 28, 18,  43, 49, 23, 37, 48, 79, 71},
        {0, 30, 121, 115, 72, 13, 20, 32, 102, 112, 88, 78, 61, 62, 53, 90, 29,
         101, 27},
        {74,  59,  16, 8, 1,   36, 75, 39,  54, 11, 97,
         113, 114, 60, 4, 111, 57, 14, 103, 81, 66, 124},
        {69,  12, 67,  7,   5,  84, 58, 83, 116, 22, 89,
         125, 21, 106, 127, 96, 42, 64, 51, 87,  9,  26},
        {40},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}}},
      {{{},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {56, 193, 23, 73, 175, 250, 139, 64, 49, 30, 31, 246, 92},
        {66, 206, 173, 129, 15, 112, 68, 19, 220, 198},
        {187, 75, 81, 44, 217, 136, 25, 142, 28, 163},
        {153, 237, 114, 200, 177, 164, 191, 185, 38,  39, 113,
         45,  109, 46,  241, 57,  196, 100, 146, 150, 35, 74,
         123, 111, 62,  165, 156, 154, 192, 201, 76},
        {43,  135, 98,  40,  124, 184, 130, 131, 89,  53,  155,
         232, 223, 151, 249, 218, 3,   197, 226, 48,  147, 63,
         78,  26,  143, 152, 251, 140, 166, 128, 199, 141},
        {51,  69,  33,  183, 52, 160, 239, 247, 0,   86,  159,
         13,  227, 50,  254, 32, 121, 21,  181, 12,  127, 145,
         245, 47,  209, 213, 58, 101, 88,  34,  212, 167},
        {202, 71,  243, 253, 59,  189, 125, 103, 138, 119, 60,
         20,  230, 195, 107, 188, 87,  229, 41,  137, 122, 149,
         158, 118, 215, 95,  182, 67,  211, 82,  11,  210},
        {235, 252, 18, 178, 54,  90,  244, 5,   219, 134, 115,
         61,  248, 9,  2,   116, 132, 84,  174, 105, 16,  108,
         102, 79,  6,  36,  22,  133, 238, 240, 234, 170},
        {216, 55,  233, 180, 110, 236, 42,  186, 221, 144, 27,  194, 157, 222,
         93,  224, 65,  176, 83,  203, 172, 225, 70,  204, 207, 214, 8},
        {190, 77, 29, 117, 255, 179, 72, 208, 168, 106, 99, 126, 80, 85, 91,
         169, 4},
        {161, 1, 162, 10, 7, 148, 120, 96, 14, 97, 37, 231},
        {242, 205, 24, 94, 228, 17, 104, 171},
        {},
        {},
        {},
        {},
        {},
        {}}},
      {{{},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {115, 173, 127, 68, 242},
        {29, 85, 88, 121, 80, 25, 41, 174, 105, 168, 74, 131, 176, 182, 154},
        {133, 126, 20,  38,  157, 250, 202, 215, 22,  136,
         0,   201, 156, 183, 79,  198, 66,  70,  189, 170},
        {86,  179, 223, 120, 16,  128, 206, 243, 204, 172, 219, 160,
         239, 224, 152, 237, 101, 205, 214, 117, 241, 217, 209, 139},
        {184, 213, 84, 110, 63,  221, 254, 125, 39,  46,  190,
         163, 90,  26, 135, 162, 69,  21,  165, 245, 186, 7,
         195, 231, 51, 23,  210, 60,  59,  194, 65,  116},
        {62,  104, 113, 36,  64,  15,  81, 32,  192, 232, 92,
         252, 253, 71,  236, 31,  153, 2,  222, 77,  137, 230,
         177, 9,   12,  13,  233, 220, 47, 193, 129, 27},
        {248, 93, 33, 56,  4,   228, 148, 61,  89, 138, 229,
         123, 17, 43, 208, 226, 52,  247, 45,  73, 44,  8,
         164, 98, 14, 72,  166, 145, 114, 141, 1,  200},
        {40,  96,  191, 196, 118, 54,  95,  151, 132, 76,  181,
         227, 130, 234, 119, 169, 58,  53,  11,  28,  140, 244,
         150, 108, 19,  240, 107, 158, 175, 238, 159, 50},
        {103, 212, 171, 124, 146, 57,  211, 87,  249, 37, 251,
         5,   187, 144, 49,  6,   91,  255, 149, 225, 42, 35,
         167, 75,  106, 218, 48,  134, 30,  147, 199, 216},
        {55,  24, 188, 143, 99,  78,  122, 109, 83,  100, 197,
         94,  67, 161, 142, 18,  10,  111, 185, 207, 82,  155,
         203, 97, 112, 246, 180, 235, 102, 34,  178, 3}}}};

  double (*func)(int) = [](int iter) { return 100000 / double(iter + 1); };
  auto [res, coloring] = test2(32, 64, 14, func, test_ranking, 100000);
  cout << "Result = " << res << '\n';
  // write_task_ranking(test_ranking);

  path cwd = current_path().parent_path();
  cwd.append("output_coloring.txt");
  ofstream output_coloring(cwd, ios::trunc);
  if (!output_coloring.is_open()) {
    cout << "file is not open" << endl;
  }
  for (auto [edge, color] : coloring) {
    auto [task_id, stage, rank_from, rank_to] = edge;
    output_coloring
        << format("\"task {}, stage {}, rank from {}, rank to {}, color {}\",",
                  task_id, stage, rank_from, rank_to, color)
        << '\n';
  }
  output_coloring.close();

  cout << '{';
  for (int i = 0; i < test_ranking.size(); ++i) {
    auto &task = test_ranking[i];
    cout << '{';
    for (int j = 0; j < task.size(); ++j) {
      auto &leaf = task[j];
      cout << '{';
      int cnt = 0;
      for (int x : leaf) {
        cnt++;
        if (cnt != leaf.size())
          cout << x << ',';
        else
          cout << x;
      }
      if (j != task.size() - 1) {
        cout << "},";
      } else {
        cout << '}';
      }
    }
    if (i == test_ranking.size() - 1) {
      cout << "}";
    } else {
      cout << "},";
    }
  }
  cout << '}';

  // for (int i = 0; i < test_rankings.size(); ++i) {
  //   auto &test_ranking = test_rankings[i];

  //   auto [task_res, task_coloring] =
  //       check_congestionless_ranking(test_ranking, i, func, 10000);

  //   // auto [e1, _] =
  //   //     count_energy(test_ranking, count_stage_energy_cota_down_to_up);
  //   // auto [e2, _] = count_energy(test_ranking, count_stage_energy);
  //   // cout << i << " " << e1 << ' ' << e2 << '\n';
  // }

  // auto [tests, colorings] = read_colored_test(5, 16, 8, "-0.5");
  // for (int i = 1000; i < 1011; ++i) {
  //   write_task_ranking(tests[i]);
  //   cout << colorings[i] << '\n';
  //   cout << "------------------------\n";
  // }

  // vector<string> alpha_list = {"-1.0", "-0.9", "-0.8", "-0.7", "-0.6",
  // "-0.5",
  //                              "-0.4", "-0.3", "-0.2", "-0.1", "0.0",  "0.1",
  //                              "0.2",  "0.3",  "0.4",  "0.5",  "0.6",  "0.7",
  //                              "0.8",  "0.9",  "1.0"};
  // vector<tuple<int, int, int>> new_tls = {
  // {4, 16, 8}, {5, 16, 8}, {6, 16, 8}, {7, 16, 8}, {8, 16, 8},
  // {2, 16, 4}, {3, 16, 4}, {4, 16, 4}, {5, 16, 4}, {6, 16, 4},
  // {7, 16, 4},
  // {8, 16, 4},
  // {2, 32, 8}};
  // vector<tuple<int, int, int>> old_tls = {
  //     {2, 8, 4},  {3, 8, 4},  {4, 8, 4},  {5, 8, 4},  {2, 16, 8}, {3, 16, 8},
  //     {4, 16, 8}, {5, 16, 8}, {6, 16, 8}, {7, 16, 8}, {8, 16, 8}, {2, 32,
  //     16}};
  // vector<tuple<int, int, int>> old_tls = {{2, 32, 16}};

  // for (auto [tasks, leaves, spines] : new_tls) {
  //   string stats_filename = format("{}_{}_{}.json", tasks, leaves, spines);
  //   cout << stats_filename << endl;
  //   unordered_map<string, vector<int>> results;
  //   for (string alpha : alpha_list) {
  //     auto res =
  //         colored_test2(tasks, leaves, spines, alpha,
  //         "new_multi_tasks_tests");
  //     results[alpha] = res;
  //   }
  //   json stat_json(results);
  //   path cwd = current_path().parent_path();
  //   cwd.append("new_multi_tasks_tests_results_scenario2");
  //   cwd.append(stats_filename);
  //   ofstream file(cwd);
  //   file << stat_json;
  //   file.close();
  // }

  // for (auto [tasks, leaves, spines] : new_tls) {
  //   string stats_filename = format("{}_{}_{}.json", tasks, leaves, spines);
  //   cout << stats_filename << endl;
  //   unordered_map<string, vector<int>> results;
  //   for (string alpha : alpha_list) {
  //     auto res =
  //         colored_test1(tasks, leaves, spines, alpha,
  //         "new_multi_tasks_tests");
  //     results[alpha] = res;
  //   }
  //   json stat_json(results);
  //   path cwd = current_path().parent_path();
  //   cwd.append("new_multi_tasks_tests_results_scenario1");
  //   cwd.append(stats_filename);
  //   ofstream file(cwd);
  //   file << stat_json;
  //   file.close();
  // }
  // cout << "------------------------\n";
  // for (auto [tasks, leaves, spines] : old_tls) {
  //   string stats_filename = format("{}_{}_{}.json", tasks, leaves, spines);
  //   cout << stats_filename << endl;
  //   unordered_map<string, vector<int>> results;
  //   for (string alpha : alpha_list) {
  //     auto res =
  //         colored_test1(tasks, leaves, spines, alpha, "multi_tasks_tests");
  //     results[alpha] = res;
  //   }
  //   json stat_json(results);
  //   path cwd = current_path().parent_path();
  //   cwd.append("multi_tasks_tests_results_scenario1");
  //   cwd.append(stats_filename);
  //   ofstream file(cwd);
  //   file << stat_json;
  //   file.close();
  // }
}
