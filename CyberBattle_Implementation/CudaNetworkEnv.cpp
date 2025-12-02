/************************************************************
 * FILE 3: CudaNetworkEnv.cpp
 * 
 * Implementation of GPU-accelerated network vulnerability 
 * simulation environment.
 ************************************************************/

#include "CudaNetworkEnv.hpp"
#include <iostream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace cyber_rl {

/////////////////////////////////////////////////////////////
// CUDA ERROR CHECKING
/////////////////////////////////////////////////////////////

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)


/////////////////////////////////////////////////////////////
// FORWARD DECLARATIONS OF CUDA KERNELS
/////////////////////////////////////////////////////////////

// Defined in cuda_kernels.cu
extern "C" {
    void launch_sir_propagation(
        const int* d_offsets,
        const int* d_neighbors,
        const int* d_state,
        const int* d_isolated,
        int* d_next_state,
        int num_nodes,
        int max_neighbors,
        float p_infect,
        float p_recover,
        curandState* d_rng_states
    );
    
    void launch_init_rng(
        curandState* d_rng_states,
        int num_nodes,
        unsigned long seed
    );
    
    void launch_copy_state(
        int* d_state,
        const int* d_next_state,
        int num_nodes
    );
}

/////////////////////////////////////////////////////////////
// GraphCSR Implementation
/////////////////////////////////////////////////////////////

GraphCSR::GraphCSR() 
    : d_offsets(nullptr)
    , d_neighbors(nullptr)
    , h_offsets(nullptr)
    , h_neighbors(nullptr)
    , num_nodes(0)
    , max_neighbors(0) 
{}

GraphCSR::~GraphCSR() {
    free_memory();
}

void GraphCSR::allocate(int nodes, int neighbors_per_node) {
    num_nodes = nodes;
    max_neighbors = neighbors_per_node;
    
    // Allocate host memory
    h_offsets  = new int[num_nodes + 1];
    h_neighbors = new int[num_nodes * max_neighbors];
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_offsets,  (num_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighbors, num_nodes * max_neighbors * sizeof(int)));
}

void GraphCSR::copy_to_device() {
    CUDA_CHECK(cudaMemcpy(d_offsets,  h_offsets,
                          (num_nodes + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighbors, h_neighbors,
                          num_nodes * max_neighbors * sizeof(int),
                          cudaMemcpyHostToDevice));
}

void GraphCSR::free_memory() {
    if (d_offsets)   cudaFree(d_offsets);
    if (d_neighbors) cudaFree(d_neighbors);
    delete[] h_offsets;
    delete[] h_neighbors;
    
    d_offsets   = nullptr;
    d_neighbors = nullptr;
    h_offsets   = nullptr;
    h_neighbors = nullptr;
}

/////////////////////////////////////////////////////////////
// NetworkState Implementation
/////////////////////////////////////////////////////////////

NetworkState::NetworkState()
    : d_state(nullptr)
    , d_next_state(nullptr)
    , d_isolated(nullptr)
    , d_agent_positions(nullptr)
    , num_nodes(0)
    , num_agents(0)
{}

NetworkState::~NetworkState() {
    free_memory();
}

void NetworkState::allocate(int nodes, int agents) {
    num_nodes  = nodes;
    num_agents = agents;
    
    // Host vectors
    h_state.assign(num_nodes, 0);
    h_isolated.assign(num_nodes, 0);
    h_agent_positions.assign(num_agents, 0);
    
    // Device memory
    CUDA_CHECK(cudaMalloc(&d_state,           num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_state,      num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_isolated,        num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_agent_positions, num_agents * sizeof(int)));
}

void NetworkState::copy_to_device() {
    CUDA_CHECK(cudaMemcpy(d_state, h_state.data(),
                          num_nodes * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_isolated, h_isolated.data(),
                          num_nodes * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_agent_positions, h_agent_positions.data(),
                          num_agents * sizeof(int),
                          cudaMemcpyHostToDevice));
}

void NetworkState::copy_to_host() {
    CUDA_CHECK(cudaMemcpy(h_state.data(), d_state,
                          num_nodes * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_isolated.data(), d_isolated,
                          num_nodes * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_agent_positions.data(), d_agent_positions,
                          num_agents * sizeof(int),
                          cudaMemcpyDeviceToHost));
}

void NetworkState::free_memory() {
    if (d_state)           cudaFree(d_state);
    if (d_next_state)      cudaFree(d_next_state);
    if (d_isolated)        cudaFree(d_isolated);
    if (d_agent_positions) cudaFree(d_agent_positions);
    
    d_state           = nullptr;
    d_next_state      = nullptr;
    d_isolated        = nullptr;
    d_agent_positions = nullptr;
}

int NetworkState::count_infected() const {
    int count = 0;
    for (int s : h_state) {
        if (s == 1) ++count;
    }
    return count;
}

float NetworkState::get_infection_ratio() const {
    if (num_nodes == 0) return 0.0f;
    return static_cast<float>(count_infected()) / static_cast<float>(num_nodes);
}

/////////////////////////////////////////////////////////////
// CudaNetworkEnv Implementation
/////////////////////////////////////////////////////////////

CudaNetworkEnv::CudaNetworkEnv(const CudaEnvConfig& config)
    : config_(config)
    , d_rng_states_(nullptr)
    , episode_step_(0)
    , prev_infected_count_(0)
{
    initialize_cuda();
    generate_network();
}

CudaNetworkEnv::~CudaNetworkEnv() {
    if (d_rng_states_) {
        cudaFree(d_rng_states_);
    }
}

void CudaNetworkEnv::initialize_cuda() {
    // Allocate graph & state
    graph_.allocate(config_.num_nodes, config_.max_neighbors);
    state_.allocate(config_.num_nodes, config_.num_agents);
    
    // RNG states
    CUDA_CHECK(cudaMalloc(&d_rng_states_,
                          config_.num_nodes * sizeof(curandState)));
    launch_init_rng(d_rng_states_, config_.num_nodes, 42UL);
    
    std::cout << "[CudaNetworkEnv] Initialized with "
              << config_.num_nodes << " nodes, "
              << config_.num_agents << " agents\n";
}

void CudaNetworkEnv::generate_network() {
    // Build hierarchical topology on host
    generate_hierarchical_network(
        graph_,
        config_.core_nodes,
        config_.dist_nodes,
        config_.num_nodes - config_.core_nodes - config_.dist_nodes,
        config_.max_neighbors
    );
    
    // Upload to GPU
    graph_.copy_to_device();
    
    std::cout << "[CudaNetworkEnv] Generated hierarchical network: "
              << config_.core_nodes  << " core, "
              << config_.dist_nodes  << " dist, "
              << (config_.num_nodes - config_.core_nodes - config_.dist_nodes)
              << " access nodes\n";
}

State CudaNetworkEnv::reset() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Clear host state
    std::fill(state_.h_state.begin(),    state_.h_state.end(),    0);
    std::fill(state_.h_isolated.begin(), state_.h_isolated.end(), 0);
    
    seed_initial_infection();
    deploy_agents();
    
    state_.copy_to_device();
    
    episode_step_        = 0;
    prev_infected_count_ = state_.count_infected();
    
    return get_agent_observation(0);
}

void CudaNetworkEnv::seed_initial_infection() {
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> core_dist(0, config_.core_nodes - 1);
    
    for (int i = 0; i < config_.initial_infected; ++i) {
        int target = core_dist(rng);
        state_.h_state[target] = 1; // infected
    }
}

void CudaNetworkEnv::deploy_agents() {
    // Place agents on highest-degree nodes
    std::vector<std::pair<int,int>> degree_nodes;
    degree_nodes.reserve(config_.num_nodes);
    
    for (int v = 0; v < config_.num_nodes; ++v) {
        int degree = graph_.h_offsets[v + 1] - graph_.h_offsets[v];
        degree_nodes.emplace_back(degree, v);
    }
    
    std::sort(degree_nodes.rbegin(), degree_nodes.rend());
    
    for (int i = 0; i < config_.num_agents; ++i) {
        state_.h_agent_positions[i] = degree_nodes[i].second;
    }
}

StepResult CudaNetworkEnv::step(int action) {
    return step_agent(0, action);
}

StepResult CudaNetworkEnv::step_agent(int agent_id, int action) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    prev_infected_count_ = state_.count_infected();
    
    apply_agent_action(agent_id, action);
    run_sir_propagation();
    sync_state_to_host();
    
    float reward = compute_reward(agent_id, action);
    
    episode_step_++;
    bool done = is_episode_done();
    
    State next_state = get_agent_observation(agent_id);
    
    StepResult result;
    result.next_state = next_state;
    result.reward     = reward;
    // If your StepResult has a single `done` bool, use that.
    // If it uses terminated/truncated, adapt this to:
    //   result.terminated = done;
    //   result.truncated  = false;
    result.done       = done;
    
    return result;
}

void CudaNetworkEnv::apply_agent_action(int agent_id, int action) {
    int current_pos = state_.h_agent_positions[agent_id];
    
    if (action < config_.move_actions) {
        // MOVE: go to neighbor[action]
        int neighbor_idx = graph_.h_offsets[current_pos] + action;
        int next_pos     = graph_.h_neighbors[neighbor_idx];
        state_.h_agent_positions[agent_id] = next_pos;
    } else if (action == config_.move_actions) {
        // SCAN: no direct state change
    } else if (action == config_.move_actions + 1) {
        // PATCH
        state_.h_state[current_pos] = 2; // Recovered / patched
    } else if (action == config_.move_actions + 2) {
        // ISOLATE
        state_.h_isolated[current_pos] = 1 - state_.h_isolated[current_pos];
    }
    
    state_.copy_to_device();
}

void CudaNetworkEnv::run_sir_propagation() {
    launch_sir_propagation(
        graph_.d_offsets,
        graph_.d_neighbors,
        state_.d_state,
        state_.d_isolated,
        state_.d_next_state,
        config_.num_nodes,
        config_.max_neighbors,
        config_.p_infect,
        config_.p_recover,
        d_rng_states_
    );
    
    launch_copy_state(state_.d_state, state_.d_next_state, config_.num_nodes);
    CUDA_CHECK(cudaDeviceSynchronize());
}

float CudaNetworkEnv::compute_reward(int agent_id, int action) {
    float reward = -0.01f; // small step penalty
    
    int current_pos = state_.h_agent_positions[agent_id];
    int node_state  = state_.h_state[current_pos];
    
    // --- Local action shaping ---
    if (action == config_.move_actions) {
        // SCAN: reward proportional to local infections seen
        int local_infected = 0;
        int start = graph_.h_offsets[current_pos];
        int end   = graph_.h_offsets[current_pos + 1];
        for (int i = start; i < end; ++i) {
            int n = graph_.h_neighbors[i];
            if (state_.h_state[n] == 1) local_infected++;
        }
        reward += 0.5f * static_cast<float>(local_infected);
    } else if (action == config_.move_actions + 1) {
        // PATCH
        if (node_state == 1) {
            reward += 10.0f; // cured infection
        } else if (node_state == 0) {
            reward += 1.0f;  // hardened vulnerable node
        }
    } else if (action == config_.move_actions + 2) {
        // ISOLATE
        if (node_state == 1 && state_.h_isolated[current_pos] == 1) {
            reward += 5.0f; // quarantined infected node
        }
    }
    
    // --- Global containment reward ---
    int current_infected      = state_.count_infected();
    int infections_prevented  = prev_infected_count_ - current_infected;
    
    if (infections_prevented > 0) {
        reward += infections_prevented * 1.0f;
    } else if (infections_prevented < 0) {
        reward += infections_prevented * 0.5f; // negative
    }
    
    // --- Tier-aware incentives ---
    if (current_pos < config_.core_nodes && node_state == 1 &&
        action == config_.move_actions + 1) {
        // Patching infected core node
        reward += 5.0f;
    }
    
    // --- Terminal bonuses ---
    if (is_success()) {
        reward += 50.0f;
    } else if (is_failure()) {
        reward -= 50.0f;
    }
    
    return reward;
}

State CudaNetworkEnv::get_agent_observation(int agent_id) const {
    int current_pos = state_.h_agent_positions[agent_id];
    
    // Local infection density
    int local_infected = 0;
    int start = graph_.h_offsets[current_pos];
    int end   = graph_.h_offsets[current_pos + 1];
    for (int i = start; i < end; ++i) {
        int n = graph_.h_neighbors[i];
        if (state_.h_state[n] == 1) local_infected++;
    }
    
    std::vector<float> features(7);
    
    // 0: normalized node index
    features[0] = static_cast<float>(current_pos) / static_cast<float>(config_.num_nodes);
    
    // 1: local infection density
    features[1] = static_cast<float>(local_infected) /
                  static_cast<float>(config_.max_neighbors);
    
    // 2: global infection ratio
    features[2] = get_infection_ratio();
    
    // 3–5: tier infection ratios
    features[3] = static_cast<float>(get_core_infected()) /
                  static_cast<float>(config_.core_nodes);
    features[4] = static_cast<float>(get_dist_infected()) /
                  static_cast<float>(config_.dist_nodes);
    features[5] = static_cast<float>(get_access_infected()) /
                  static_cast<float>(config_.num_nodes - config_.core_nodes - config_.dist_nodes);
    
    // 6: tier indicator (0 = access, 0.5 = dist, 1 = core)
    float tier_val = 0.0f;
    if (current_pos < config_.core_nodes) {
        tier_val = 1.0f;
    } else if (current_pos < config_.core_nodes + config_.dist_nodes) {
        tier_val = 0.5f;
    }
    features[6] = tier_val;
    
    return State(features, current_pos);
}

std::vector<float> CudaNetworkEnv::build_observation(int agent_id) const {
    return get_agent_observation(agent_id).features;
}

void CudaNetworkEnv::sync_state_to_host() {
    state_.copy_to_host();
}

void CudaNetworkEnv::sync_state_to_device() {
    state_.copy_to_device();
}

int CudaNetworkEnv::get_observation_dim() const {
    return 7; // matches features size above
}

int CudaNetworkEnv::get_action_dim() const {
    return config_.move_actions + config_.defense_actions; // 11
}

std::string CudaNetworkEnv::get_name() const {
    return "CudaNetworkEnv-" + std::to_string(config_.num_nodes);
}

float CudaNetworkEnv::get_infection_ratio() const {
    return state_.get_infection_ratio();
}

int CudaNetworkEnv::get_infected_count() const {
    return state_.count_infected();
}

int CudaNetworkEnv::get_core_infected() const {
    int count = 0;
    for (int i = 0; i < config_.core_nodes; ++i) {
        if (state_.h_state[i] == 1) ++count;
    }
    return count;
}

int CudaNetworkEnv::get_dist_infected() const {
    int count = 0;
    int start = config_.core_nodes;
    int end   = config_.core_nodes + config_.dist_nodes;
    for (int i = start; i < end; ++i) {
        if (state_.h_state[i] == 1) ++count;
    }
    return count;
}

int CudaNetworkEnv::get_access_infected() const {
    int count = 0;
    int start = config_.core_nodes + config_.dist_nodes;
    int end   = config_.num_nodes;
    for (int i = start; i < end; ++i) {
        if (state_.h_state[i] == 1) ++count;
    }
    return count;
}

bool CudaNetworkEnv::is_episode_done() const {
    return is_success() || is_failure() ||
           (episode_step_ >= config_.max_episode_steps);
}

bool CudaNetworkEnv::is_success() const {
    return (episode_step_ > 100) &&
           (get_infection_ratio() < config_.success_threshold);
}

bool CudaNetworkEnv::is_failure() const {
    return get_infection_ratio() > config_.failure_threshold;
}

void CudaNetworkEnv::infect_node(int node_id) {
    if (node_id < 0 || node_id >= config_.num_nodes) return;
    state_.h_state[node_id] = 1;
    sync_state_to_device();
}

void CudaNetworkEnv::recover_node(int node_id) {
    if (node_id < 0 || node_id >= config_.num_nodes) return;
    state_.h_state[node_id] = 2;
    sync_state_to_device();
}

void CudaNetworkEnv::isolate_node(int node_id) {
    if (node_id < 0 || node_id >= config_.num_nodes) return;
    state_.h_isolated[node_id] = 1;
    sync_state_to_device();
}

/////////////////////////////////////////////////////////////
// Graph Generation Utility
/////////////////////////////////////////////////////////////

void generate_hierarchical_network(
    GraphCSR& graph,
    int core_nodes,
    int dist_nodes,
    int access_nodes,
    int max_neighbors)
{
    std::mt19937 rng(42);
    int total_nodes = core_nodes + dist_nodes + access_nodes;
    
    std::vector<std::vector<int>> adj(total_nodes);
    
    // Core: dense mesh up to max_neighbors
    for (int i = 0; i < core_nodes; ++i) {
        for (int j = 0; j < core_nodes; ++j) {
            if (i != j && static_cast<int>(adj[i].size()) < max_neighbors) {
                adj[i].push_back(j);
            }
        }
    }
    
    // Dist layer: connect to core + peers
    for (int i = core_nodes; i < core_nodes + dist_nodes; ++i) {
        std::uniform_int_distribution<int> core_dist(0, core_nodes - 1);
        for (int k = 0; k < 3 && static_cast<int>(adj[i].size()) < max_neighbors; ++k) {
            adj[i].push_back(core_dist(rng));
        }
        
        std::uniform_int_distribution<int> peer_dist(core_nodes,
                                                     core_nodes + dist_nodes - 1);
        for (int k = 0; k < 2 && static_cast<int>(adj[i].size()) < max_neighbors; ++k) {
            int peer = peer_dist(rng);
            if (peer != i) adj[i].push_back(peer);
        }
    }
    
    // Access layer: connect to 1–2 dist nodes
    for (int i = core_nodes + dist_nodes; i < total_nodes; ++i) {
        std::uniform_int_distribution<int> dist_dist(core_nodes,
                                                     core_nodes + dist_nodes - 1);
        adj[i].push_back(dist_dist(rng));
        
        if ((rng() & 1) && static_cast<int>(adj[i].size()) < max_neighbors) {
            adj[i].push_back(dist_dist(rng));
        }
    }
    
    // Pad with self-loops
    for (int v = 0; v < total_nodes; ++v) {
        while (static_cast<int>(adj[v].size()) < max_neighbors) {
            adj[v].push_back(v);
        }
    }
    
    // Build CSR
    int offset = 0;
    for (int v = 0; v < total_nodes; ++v) {
        graph.h_offsets[v] = offset;
        for (int u : adj[v]) {
            graph.h_neighbors[offset++] = u;
        }
    }
    graph.h_offsets[total_nodes] = offset;
}

} // namespace cyber_rl