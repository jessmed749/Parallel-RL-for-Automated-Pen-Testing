/************************************************************
 *  SECTION 1 — GRAPH + STATE STRUCTURES
 *  CPU/GPU Hybrid RL Environment for 50k Node Network
 *  SIR contagion + Soft Isolation + Q-learning (Option B)
 ************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <vector>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>

/////////////////////////////////////////////////////////////
// CONFIGURATION
/////////////////////////////////////////////////////////////

static const int NUM_NODES       = 50000;
static const int MAX_NEIGHBORS   = 8;      // movement actions
static const int EXTRA_ACTIONS   = 3;      // scan, patch, isolate
static const int TOTAL_ACTIONS   = MAX_NEIGHBORS + EXTRA_ACTIONS;

static const int START_INFECTED  = 20;     // initial infected nodes

// SIR parameters
static const float P_INFECT   = 0.03f;     // infection probability per edge
static const float P_RECOVER  = 0.01f;     // recovery probability per step

// RL step penalties / rewards (tunable later)
static const float REWARD_PATCH      = +1.0f;
static const float REWARD_SCAN       = +0.1f;
static const float REWARD_ISOLATE    = +0.25f;
static const float REWARD_INFECTED   = -0.05f;
static const float REWARD_STEP       = -0.001f;

// Episode end thresholds
static const float MAX_INFECTED_RATIO = 0.25f;  // fail if >25% infected

// NEW: Missing constants
static const int NUM_WORKERS = 4;           // Number of worker threads
static const int BATCH_SIZE = 256;          // Q-learning batch size
static const int MIN_REPLAY_SIZE = 1000;    // Min samples before training
static const int REPLAY_CAP = 100000;       // Replay buffer capacity
static const int MAX_EPISODE_STEPS = 500;   // Episode length limit
static const float EPSILON = 0.1f;          // Exploration rate
static const float ALPHA = 0.01f;           // Q-learning rate
static const float GAMMA = 0.99f;           // Discount factor


/////////////////////////////////////////////////////////////
// DEVICE UTIL MACROS
/////////////////////////////////////////////////////////////

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    }


/////////////////////////////////////////////////////////////
// CSR GRAPH STRUCTURE
/////////////////////////////////////////////////////////////

struct GraphCSR {
    int    *d_offsets;       // size: NUM_NODES+1
    int    *d_neighbors;     // size: NUM_NODES * MAX_NEIGHBORS

    int    *h_offsets;
    int    *h_neighbors;

    GraphCSR() : d_offsets(nullptr), d_neighbors(nullptr),
                 h_offsets(nullptr), h_neighbors(nullptr) {}
};


/////////////////////////////////////////////////////////////
// SIR + ISOLATION DEVICE STATE ARRAYS
/////////////////////////////////////////////////////////////

// 0 = Susceptible
// 1 = Infected
// 2 = Recovered
//
// isolated[v] = 1 means the node cannot get infected or infect others.

struct DeviceState {
    int *d_state;          // SIR state of each node
    int *d_next_state;     // buffer for contagion kernel
    int *d_isolated;       // bool array

    DeviceState() : d_state(nullptr), d_next_state(nullptr), d_isolated(nullptr) {}
};


/////////////////////////////////////////////////////////////
// HOST STATE
/////////////////////////////////////////////////////////////

struct HostState {
    std::vector<int> state;       // SIR
    std::vector<int> isolated;    // soft isolation

    HostState() {
        state.resize(NUM_NODES);
        isolated.resize(NUM_NODES);
    }
};


/////////////////////////////////////////////////////////////
// GRAPH GENERATION — REACHABLE, SCALE-FREE, FORWARD EDGES
/////////////////////////////////////////////////////////////

static void generate_graph(GraphCSR &graph)
{
    graph.h_offsets   = new int[NUM_NODES + 1];
    graph.h_neighbors = new int[NUM_NODES * MAX_NEIGHBORS];

    std::mt19937 rng(42);

    std::vector<std::vector<int>> adj(NUM_NODES);

    // Preferential attachment — classic Barabási–Albert style
    std::vector<int> degree(NUM_NODES, 0);

    for (int v = 1; v < NUM_NODES; v++) {
        // Choose some backward edges to older nodes
        int edges_to_add = std::min(MAX_NEIGHBORS - 2, v);

        // Preferential selection over previous nodes
        std::unordered_set<int> chosen;
        std::uniform_int_distribution<int> dist_prev(0, v - 1);

        for (int e = 0; e < edges_to_add; e++) {
            int u = dist_prev(rng);
            chosen.insert(u);
        }

        // Add them
        for (int u : chosen) {
            adj[v].push_back(u);
            degree[v]++;
            degree[u]++;
        }

        // Add a few random *forward* edges to ensure reachability
        std::uniform_int_distribution<int> dist_forward(v + 1, NUM_NODES - 1);
        int fwd_edges = 1;   // 1 forward edge guarantees upward reach

        if (v < NUM_NODES - 1) {
            for (int i = 0; i < fwd_edges; i++) {
                int f = dist_forward(rng);
                adj[v].push_back(f);
                degree[v]++;
                degree[f]++;
            }
        }

        // If neighborhood < MAX_NEIGHBORS, add uniform random edges
        while ((int)adj[v].size() < MAX_NEIGHBORS) {
            std::uniform_int_distribution<int> dist_all(0, NUM_NODES - 1);
            int r = dist_all(rng);
            if (r != v)
                adj[v].push_back(r);
        }
    }

    // Build CSR
    int offset = 0;
    for (int v = 0; v < NUM_NODES; v++) {
        graph.h_offsets[v] = offset;
        for (int u : adj[v]) {
            graph.h_neighbors[offset++] = u;
        }
        // Ensure each row has exactly MAX_NEIGHBORS entries
        while (offset < graph.h_offsets[v] + MAX_NEIGHBORS) {
            graph.h_neighbors[offset++] = 0;
        }
    }
    graph.h_offsets[NUM_NODES] = offset;

    // Copy to device
    CUDA_CHECK(cudaMalloc(&graph.d_offsets,   sizeof(int) * (NUM_NODES + 1)));
    CUDA_CHECK(cudaMalloc(&graph.d_neighbors, sizeof(int) * (NUM_NODES * MAX_NEIGHBORS)));

    CUDA_CHECK(cudaMemcpy(graph.d_offsets,   graph.h_offsets,
                          sizeof(int)*(NUM_NODES+1),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(graph.d_neighbors, graph.h_neighbors,
                          sizeof(int)*(NUM_NODES*MAX_NEIGHBORS),
                          cudaMemcpyHostToDevice));

    printf("Graph generated: %d nodes, %d neighbors per node\n",
           NUM_NODES, MAX_NEIGHBORS);
}


/////////////////////////////////////////////////////////////
// HOST + DEVICE STATE INITIALIZATION
/////////////////////////////////////////////////////////////

static void init_host_state(HostState &host)
{
    std::fill(host.state.begin(), host.state.end(), 0);     // S
    std::fill(host.isolated.begin(), host.isolated.end(), 0);

    std::mt19937 rng(123);

    // randomly infect START_INFECTED nodes
    for (int i = 0; i < START_INFECTED; i++) {
        int v = rng() % NUM_NODES;
        host.state[v] = 1; // Infected
    }
}

static void init_device_state(DeviceState &dev, const HostState &host)
{
    CUDA_CHECK(cudaMalloc(&dev.d_state,       NUM_NODES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev.d_next_state,  NUM_NODES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev.d_isolated,    NUM_NODES * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dev.d_state, host.state.data(),
                          NUM_NODES * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dev.d_isolated, host.isolated.data(),
                          NUM_NODES * sizeof(int),
                          cudaMemcpyHostToDevice));
}

/////////////////////////////////////////////////////////////
// DEBUG TOOLS
/////////////////////////////////////////////////////////////

static void print_node_info(const HostState &host, int node)
{
    printf("Node %d: SIR=%d iso=%d\n",
        node, host.state[node], host.isolated[node]);
}

static float compute_infected_ratio(const HostState &host)
{
    int cnt = 0;
    for (int i = 0; i < NUM_NODES; i++)
        cnt += (host.state[i] == 1);
    return float(cnt) / float(NUM_NODES);
}

/************************************************************
 *  SECTION 2 — GPU KERNELS
 *  SIR contagion + Isolation + Q-learning updates
 ************************************************************/

/////////////////////////////////////////////////////////////
// 1. GPU KERNEL — SIR CONTAGION WITH ISOLATION
/////////////////////////////////////////////////////////////

__global__ void sir_contagion_step(
        const int * __restrict__ d_offsets,
        const int * __restrict__ d_neighbors,
        const int * __restrict__ d_state,
        const int * __restrict__ d_isolated,
        int *d_next_state,
        int num_nodes,
        float p_infect,
        float p_recover,
        curandState *rng_states)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;

    int s = d_state[v];
    int iso = d_isolated[v];

    // Copy state by default
    int new_state = s;

    // Recovered stays recovered
    if (s == 2) {
        d_next_state[v] = 2;
        return;
    }

    curandState local_rng = rng_states[v];

    // INFECTED → RECOVER
    if (s == 1)
    {
        float r = curand_uniform(&local_rng);
        if (r < p_recover)
            new_state = 2;   // recovered
        else
            new_state = 1;   // remain infected
    }

    // SUSCEPTIBLE → possible infection
    if (s == 0)
    {
        if (!iso)    // cannot infect or be infected while isolated
        {
            int start = d_offsets[v];
            for (int i = 0; i < MAX_NEIGHBORS; i++)
            {
                int u = d_neighbors[start + i];
                int sus_neighbor_state = d_state[u];
                int sus_neighbor_iso   = d_isolated[u];

                if (sus_neighbor_state == 1 && !sus_neighbor_iso)
                {
                    float r = curand_uniform(&local_rng);
                    if (r < p_infect) {
                        new_state = 1;
                        break;
                    }
                }
            }
        }
    }

    rng_states[v] = local_rng;
    d_next_state[v] = new_state;
}

/////////////////////////////////////////////////////////////
// RNG INITIALIZATION KERNEL
/////////////////////////////////////////////////////////////

__global__ void init_rng_kernel(curandState *states, int n, unsigned long seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        curand_init(seed, id, 0, &states[id]);
}

/////////////////////////////////////////////////////////////
// 2. GPU KERNEL — APPLY ACTION EFFECTS
//    Action mapping:
//    0..7 = move to neighbor slot
//    8    = SCAN      (no state change)
//    9    = PATCH     (force SIR = Recovered)
//    10   = ISOLATE   (toggle isolation flag)
/////////////////////////////////////////////////////////////

__global__ void apply_action_kernel(
        const int *d_offsets,
        const int *d_neighbors,
        int *d_state,
        int *d_isolated,
        int *d_agent_pos,
        const int *d_actions,
        int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    int node = d_agent_pos[i];
    int action = d_actions[i];

    if (action < MAX_NEIGHBORS)
    {
        // MOVE to neighbor
        int start = d_offsets[node];
        int next = d_neighbors[start + action];
        d_agent_pos[i] = next;
        return;
    }

    // SCAN
    if (action == MAX_NEIGHBORS) {
        // no state change — reward handled on CPU
        return;
    }

    // PATCH — cure infection, set to recovered
    if (action == MAX_NEIGHBORS + 1) {
        d_state[node] = 2;    // Recovered
        return;
    }

    // ISOLATE — toggle soft isolation
    if (action == MAX_NEIGHBORS + 2) {
        d_isolated[node] = (d_isolated[node] == 0 ? 1 : 0);
        return;
    }
}

/////////////////////////////////////////////////////////////
// 3. GPU KERNEL — COPY NEXT_STATE → STATE
/////////////////////////////////////////////////////////////

__global__ void copy_next_state_kernel(
        int *d_state,
        const int *d_next_state,
        int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d_state[i] = d_next_state[i];
}

/////////////////////////////////////////////////////////////
// 4. GPU KERNEL — Q-LEARNING UPDATE
//    Simple tabular Q-learning:
//    Q[s][a] = Q + α * (r + γ * max_a' Q[s'][a'] - Q)
/////////////////////////////////////////////////////////////

__global__ void q_update_kernel(
        const int *d_states,
        const int *d_actions,
        const float *d_rewards,
        const int *d_next_states,
        float *d_Q,
        int batch_size,
        int num_actions,
        float alpha,
        float gamma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    int s  = d_states[i];
    int a  = d_actions[i];
    int sn = d_next_states[i];
    float r = d_rewards[i];

    float best_next = -1e9f;
    int base = sn * num_actions;

    // max Q(s', a')
    #pragma unroll
    for (int j = 0; j < MAX_NEIGHBORS + EXTRA_ACTIONS; j++)
    {
        float q = d_Q[base + j];
        if (q > best_next)
            best_next = q;
    }

    // TD update
    int idx = s * num_actions + a;
    float old_q = d_Q[idx];
    float new_q = old_q + alpha * (r + gamma * best_next - old_q);
    d_Q[idx] = new_q;
}

/************************************************************
 *  SECTION 3 — CPU ENVIRONMENT
 *  Worker threads, replay buffer, Q-table updates
 ************************************************************/

//----------------------------------------
// 1. Transition struct for replay
//----------------------------------------
struct Transition {
    int state;
    int action;
    float reward;
    int next_state;
    bool done;
};

//----------------------------------------
// 2. Replay buffer
//----------------------------------------
class ReplayBuffer {
public:
    ReplayBuffer(size_t capacity) 
        : buffer_(capacity), write_pos_(0), size_(0) {}

    void push(const Transition &t) {
        size_t pos = write_pos_.fetch_add(1) % buffer_.size();
        buffer_[pos] = t;
        size_.store(std::min(buffer_.size(), size_.load() + 1));
    }

    size_t size() const { return size_.load(); }

    // Sample batch of transitions
    std::vector<Transition> sample(size_t batch_size, std::mt19937 &rng) {
        std::vector<Transition> out;
        size_t n = std::min(batch_size, size());
        std::uniform_int_distribution<size_t> dist(0, size() - 1);
        out.reserve(n);
        for (size_t i = 0; i < n; i++)
            out.push_back(buffer_[dist(rng)]);
        return out;
    }

private:
    std::vector<Transition> buffer_;
    std::atomic<size_t> write_pos_;
    std::atomic<size_t> size_;
};

//----------------------------------------
// 3. Metrics
//----------------------------------------
struct Metrics {
    std::atomic<int> episodes_completed{0};
    std::atomic<int> targets_reached{0};
    std::atomic<long long> total_steps{0};
    std::atomic<int> episode_timeouts{0};
};

//----------------------------------------
// 4. Worker thread
//----------------------------------------
void worker_thread(
    int id,
    const int* row_offsets,
    const int* col_indices,
    std::vector<int> &state,
    std::vector<int> &isolated,
    ReplayBuffer &replay,
    std::atomic<bool> &stop_flag,
    Metrics &metrics,
    const std::vector<float> &h_Q,
    std::mutex &state_mutex)
{
    std::mt19937 rng(id + std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::uniform_int_distribution<int> act_dist(0, MAX_NEIGHBORS + EXTRA_ACTIONS - 1);
    std::uniform_int_distribution<int> node_dist(0, NUM_NODES - 1);

    int agent_state = node_dist(rng);
    int episode_step = 0;
    float episode_reward = 0.0f;

    while (!stop_flag.load()) {

        // Epsilon-greedy with Q-table
        int action = 0;
        if (uni(rng) < EPSILON) {
            action = act_dist(rng);
        } else {
            // Greedy: select best Q
            int best_action = 0;
            float best_q = -1e9f;
            for (int a = 0; a < MAX_NEIGHBORS + EXTRA_ACTIONS; a++) {
                float q = h_Q[agent_state * (MAX_NEIGHBORS + EXTRA_ACTIONS) + a];
                if (q > best_q) {
                    best_q = q;
                    best_action = a;
                }
            }
            action = best_action;
        }

        // Apply action (simplified CPU version)
        int next_state = agent_state;
        float reward = -0.01f;  // small step penalty
        bool done = false;

        if (action < MAX_NEIGHBORS) {
            int start = row_offsets[agent_state];
            next_state = col_indices[start + action];
        } else if (action == MAX_NEIGHBORS + 1) {
            std::lock_guard<std::mutex> lock(state_mutex);
            state[agent_state] = 2;  // PATCH -> recovered
        } else if (action == MAX_NEIGHBORS + 2) {
            std::lock_guard<std::mutex> lock(state_mutex);
            isolated[agent_state] = 1 - isolated[agent_state]; // toggle
        }

        // Reward shaping: reaching high-index nodes
        if (next_state >= NUM_NODES - 100) {
            reward = 10.0f;
            done = true;
            metrics.targets_reached++;
        }

        episode_step++;
        if (episode_step >= MAX_EPISODE_STEPS && !done) {
            done = true;
            metrics.episode_timeouts++;
        }

        episode_reward += reward;

        // Push to replay buffer
        replay.push({agent_state, action, reward, next_state, done});
        metrics.total_steps++;

        if (done) {
            metrics.episodes_completed++;
            agent_state = node_dist(rng);
            episode_step = 0;
            episode_reward = 0.0f;
        } else {
            agent_state = next_state;
        }
    }
}

//----------------------------------------
// 5. CPU loop to prepare batches for GPU update
//----------------------------------------
void train_loop(
    ReplayBuffer &replay,
    float *d_Q,
    int num_actions,
    curandState *rng_states,
    std::atomic<bool> &stop_flag)
{
    std::mt19937 sample_rng(42);

    // Device memory for batch
    int *d_states, *d_actions, *d_next_states;
    float *d_rewards;
    
    CUDA_CHECK(cudaMalloc(&d_states, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_actions, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_states, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rewards, BATCH_SIZE * sizeof(float)));

    for (int iter = 0; iter < 5000 && !stop_flag.load(); iter++) {
        if (replay.size() < MIN_REPLAY_SIZE) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        auto batch = replay.sample(BATCH_SIZE, sample_rng);

        std::vector<int> h_states(BATCH_SIZE);
        std::vector<int> h_actions(BATCH_SIZE);
        std::vector<int> h_next_states(BATCH_SIZE);
        std::vector<float> h_rewards(BATCH_SIZE);

        for (size_t i = 0; i < batch.size(); i++) {
            h_states[i]      = batch[i].state;
            h_actions[i]     = batch[i].action;
            h_next_states[i] = batch[i].next_state;
            h_rewards[i]     = batch[i].reward;
        }

        // Copy to GPU
        CUDA_CHECK(cudaMemcpy(d_states, h_states.data(), 
                   BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_actions, h_actions.data(), 
                   BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_next_states, h_next_states.data(), 
                   BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rewards, h_rewards.data(), 
                   BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));

        // Launch Q-learning kernel
        int threads = 256;
        int blocks = (BATCH_SIZE + threads - 1) / threads;
        q_update_kernel<<<blocks, threads>>>(
            d_states, d_actions, d_rewards, d_next_states,
            d_Q, BATCH_SIZE, num_actions, ALPHA, GAMMA);
        
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Cleanup
    cudaFree(d_states);
    cudaFree(d_actions);
    cudaFree(d_next_states);
    cudaFree(d_rewards);
}

/************************************************************
 *  SECTION 4 — MAIN FUNCTION
 *  Initialization, workers, GPU SIR, training loop, eval
 ************************************************************/

int main() {
    std::cout << "=== INITIALIZATION ===\n";

    // 1. Generate graph using existing function
    GraphCSR graph;
    generate_graph(graph);

    // 2. Initialize CPU state arrays
    std::vector<int> state(NUM_NODES, 0);       // 0=S,1=I,2=R
    std::vector<int> next_state(NUM_NODES, 0);
    std::vector<int> isolated(NUM_NODES, 0);

    std::mt19937 rng(42);
    // Infect a few random nodes
    for (int i = 0; i < START_INFECTED; i++)
        state[rng() % NUM_NODES] = 1;

    // 3. Allocate GPU memory
    int *d_state, *d_next_state, *d_isolated;
    CUDA_CHECK(cudaMalloc(&d_state, NUM_NODES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_state, NUM_NODES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_isolated, NUM_NODES * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_state, state.data(), NUM_NODES * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_isolated, isolated.data(), NUM_NODES * sizeof(int), cudaMemcpyHostToDevice));

    // Q-table
    float *d_Q;
    CUDA_CHECK(cudaMalloc(&d_Q, NUM_NODES * (MAX_NEIGHBORS + EXTRA_ACTIONS) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_Q, 0, NUM_NODES * (MAX_NEIGHBORS + EXTRA_ACTIONS) * sizeof(float)));

    // Host copy of Q-table for workers
    std::vector<float> h_Q(NUM_NODES * (MAX_NEIGHBORS + EXTRA_ACTIONS), 0.0f);

    // RNG states
    curandState *d_rng_states;
    CUDA_CHECK(cudaMalloc(&d_rng_states, NUM_NODES * sizeof(curandState)));
    init_rng_kernel<<<(NUM_NODES + 255)/256, 256>>>(d_rng_states, NUM_NODES, 1234);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. Replay buffer and metrics
    ReplayBuffer replay(REPLAY_CAP);
    Metrics metrics;
    std::atomic<bool> stop_flag(false);
    std::mutex state_mutex;

    // 5. Launch worker threads
    std::vector<std::thread> workers;
    for (int i = 0; i < NUM_WORKERS; ++i)
        workers.emplace_back(worker_thread, i, 
                             graph.h_offsets,
                             graph.h_neighbors, 
                             std::ref(state),
                             std::ref(isolated), 
                             std::ref(replay),
                             std::ref(stop_flag), 
                             std::ref(metrics),
                             std::cref(h_Q), 
                             std::ref(state_mutex));

    // 6. Launch training thread
    std::thread trainer(train_loop, std::ref(replay), d_Q, 
                       MAX_NEIGHBORS + EXTRA_ACTIONS, d_rng_states, 
                       std::ref(stop_flag));

    // 7. Main training loop
    std::cout << "=== TRAINING ===\n";

    auto start_time = std::chrono::steady_clock::now();
    for (int iter = 0; iter < 5000; ++iter) {

        // 7a. GPU contagion step
        int threads = 256;
        int blocks = (NUM_NODES + threads - 1) / threads;
        sir_contagion_step<<<blocks, threads>>>(
            graph.d_offsets, graph.d_neighbors, d_state,
            d_isolated, d_next_state,
            NUM_NODES, P_INFECT, P_RECOVER, d_rng_states);
        copy_next_state_kernel<<<blocks, threads>>>(d_state, d_next_state, NUM_NODES);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 7b. Periodically update host Q-table for workers
        if (iter % 50 == 0) {
            CUDA_CHECK(cudaMemcpy(h_Q.data(), d_Q, 
                       h_Q.size() * sizeof(float), 
                       cudaMemcpyDeviceToHost));
        }

        // 7c. Logging every 300 iterations
        if (iter % 300 == 0) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            auto sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            int episodes = metrics.episodes_completed.load();
            int targets = metrics.targets_reached.load();
            int timeouts = metrics.episode_timeouts.load();
            long long steps = metrics.total_steps.load();
            float success_rate = episodes > 0 ? (100.0f * targets / episodes) : 0.0f;

            std::cout << "Iter " << iter << " [" << sec << "s]\n";
            std::cout << "  Episodes: " << episodes << " | Targets: " << targets
                      << " | Timeouts: " << timeouts
                      << " | Success: " << success_rate << "% | Steps: " << steps << "\n";
        }
    }

    // 8. Stop workers and trainer
    stop_flag.store(true);
    for (auto &t : workers) t.join();
    trainer.join();

    // 9. Final Q-table copy for evaluation
    CUDA_CHECK(cudaMemcpy(h_Q.data(), d_Q, 
               h_Q.size() * sizeof(float), 
               cudaMemcpyDeviceToHost));

    // 10. Policy evaluation (CPU)
    std::cout << "\n=== POLICY EVALUATION ===\n";
    for (int start_node : {0, 1000, 25000}) {
        int agent = start_node;
        bool reached = false;
        for (int step = 0; step < 200; ++step) {
            // Use learned Q-table
            int best_a = 0;
            float best_q = -1e9f;
            for (int a = 0; a < MAX_NEIGHBORS + EXTRA_ACTIONS; a++) {
                float q = h_Q[agent * (MAX_NEIGHBORS + EXTRA_ACTIONS) + a];
                if (q > best_q) {
                    best_q = q;
                    best_a = a;
                }
            }
            
            // Take best action (movement only for eval)
            if (best_a < MAX_NEIGHBORS) {
                int start = graph.h_offsets[agent];
                int next = graph.h_neighbors[start + best_a];
                agent = next;
            }
            
            if (agent >= NUM_NODES - 100) {
                std::cout << "Reached target from " << start_node << " in " << step+1 << " steps!\n";
                reached = true;
                break;
            }
        }
        if (!reached) std::cout << "Failed to reach target from " << start_node << "\n";
    }

    // 11. Free GPU memory
    cudaFree(d_state);
    cudaFree(d_next_state);
    cudaFree(d_isolated);
    cudaFree(d_Q);
    cudaFree(d_rng_states);

    // 12. Free graph memory
    delete[] graph.h_offsets;
    delete[] graph.h_neighbors;
    cudaFree(graph.d_offsets);
    cudaFree(graph.d_neighbors);

    std::cout << "\n=== TRAINING COMPLETE ===\n";
    std::cout << "Final Statistics:\n";
    std::cout << "  Total Episodes: " << metrics.episodes_completed.load() << "\n";
    std::cout << "  Targets Reached: " << metrics.targets_reached.load() << "\n";
    std::cout << "  Total Steps: " << metrics.total_steps.load() << "\n";
    float final_success = metrics.episodes_completed.load() > 0 ? 
        (100.0f * metrics.targets_reached.load() / metrics.episodes_completed.load()) : 0.0f;
    std::cout << "  Final Success Rate: " << final_success << "%\n";

    return 0;
}