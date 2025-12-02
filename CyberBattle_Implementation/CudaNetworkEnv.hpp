/************************************************************
 * FILE 2: CudaNetworkEnv.hpp
 *
 * GPU-accelerated large-scale network vulnerability simulation.
 * Wraps CUDA kernels for SIR propagation into IEnv interface.
 ************************************************************/

#ifndef CUDA_NETWORK_ENV_HPP
#define CUDA_NETWORK_ENV_HPP

#include "IEnv.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <mutex>

namespace cyber_rl {

/////////////////////////////////////////////////////////////
// CONFIGURATION
/////////////////////////////////////////////////////////////

struct CudaEnvConfig {
    // Network topology
    int num_nodes = 50000;
    int core_nodes = 100;
    int dist_nodes = 1000;
    int max_neighbors = 8;

    // SIR parameters
    float p_infect = 0.003f;
    float p_recover = 0.01f;
    int initial_infected = 20;

    // Defense agents
    int num_agents = 8;

    // Episode termination
    int max_episode_steps = 500;
    float failure_threshold = 0.50f;   // Fail if > 50% infected
    float success_threshold = 0.10f;   // Success if < 10% infected

    // Action space
    int move_actions = 8;              // 8 directional moves
    int defense_actions = 3;           // SCAN, PATCH, ISOLATE
};


/////////////////////////////////////////////////////////////
// GRAPH DATA STRUCTURE (CSR format)
/////////////////////////////////////////////////////////////

struct GraphCSR {
    // Device (GPU)
    int *d_offsets;
    int *d_neighbors;

    // Host (CPU)
    int *h_offsets;
    int *h_neighbors;

    int num_nodes;
    int max_neighbors;

    GraphCSR();
    ~GraphCSR();

    void allocate(int nodes, int neighbors_per_node);
    void copy_to_device();
    void free_memory();
};


/////////////////////////////////////////////////////////////
// NETWORK STATE
/////////////////////////////////////////////////////////////

struct NetworkState {
    // Device memory
    int *d_state;
    int *d_next_state;
    int *d_isolated;
    int *d_agent_positions;

    // Host memory
    std::vector<int> h_state;
    std::vector<int> h_isolated;
    std::vector<int> h_agent_positions;

    int num_nodes;
    int num_agents;

    NetworkState();
    ~NetworkState();

    void allocate(int nodes, int agents);
    void copy_to_device();
    void copy_to_host();
    void free_memory();

    int count_infected() const;
    float get_infection_ratio() const;
};


/////////////////////////////////////////////////////////////
// CUDA NETWORK ENVIRONMENT
/////////////////////////////////////////////////////////////

class CudaNetworkEnv : public IEnv {
public:
    explicit CudaNetworkEnv(const CudaEnvConfig& config = CudaEnvConfig());
    ~CudaNetworkEnv() override;

    // Disable copying
    CudaNetworkEnv(const CudaNetworkEnv&) = delete;
    CudaNetworkEnv& operator=(const CudaNetworkEnv&) = delete;

    //--------------------------------------------------------
    // IEnv Interface
    //--------------------------------------------------------

    State reset() override;
    StepResult step(int action) override;

    int get_observation_dim() const override; // defined in .cpp
    int get_action_dim() const override;      // defined in .cpp
    std::string get_name() const override;    // defined in .cpp

    //--------------------------------------------------------
    // Multi-Agent Support
    //--------------------------------------------------------

    State get_agent_observation(int agent_id) const;
    StepResult step_agent(int agent_id, int action);

    int get_num_agents() const { return config_.num_agents; }

    // NEW: allow analysis code to query actual network size
    int get_num_nodes() const { return config_.num_nodes; }

    //--------------------------------------------------------
    // Infection Statistics
    //--------------------------------------------------------

    float get_infection_ratio() const;
    int get_infected_count() const;

    int get_core_infected() const;
    int get_dist_infected() const;
    int get_access_infected() const;

    //--------------------------------------------------------
    // Termination Conditions
    //--------------------------------------------------------

    bool is_episode_done() const;
    bool is_success() const;
    bool is_failure() const;

    //--------------------------------------------------------
    // Manual Control (testing)
    //--------------------------------------------------------

    void infect_node(int node_id);
    void recover_node(int node_id);
    void isolate_node(int node_id);

private:

    //--------------------------------------------------------
    // Initialization
    //--------------------------------------------------------

    void initialize_cuda();
    void generate_network();
    void seed_initial_infection();
    void deploy_agents();

    //--------------------------------------------------------
    // GPU Simulation
    //--------------------------------------------------------

    void run_sir_propagation();
    void apply_agent_action(int agent_id, int action);

    //--------------------------------------------------------
    // Synchronization
    //--------------------------------------------------------

    void sync_state_to_host();
    void sync_state_to_device();

    //--------------------------------------------------------
    // Reward + Observation
    //--------------------------------------------------------

    float compute_reward(int agent_id, int action);
    std::vector<float> build_observation(int agent_id) const;

    //--------------------------------------------------------
    // Members
    //--------------------------------------------------------

    CudaEnvConfig config_;
    GraphCSR graph_;
    NetworkState state_;

    curandState* d_rng_states_;

    int episode_step_;
    int prev_infected_count_;

    mutable std::mutex state_mutex_;
};


/////////////////////////////////////////////////////////////
// GRAPH GENERATION UTILITY
/////////////////////////////////////////////////////////////

void generate_hierarchical_network(
    GraphCSR& graph,
    int core_nodes,
    int dist_nodes,
    int access_nodes,
    int max_neighbors
);

} // namespace cyber_rl

#endif // CUDA_NETWORK_ENV_HPP
