/************************************************************
 * FILE 7: TabularQLearning.hpp
 * 
 * Environment-agnostic tabular Q-learning algorithm.
 * Works with any IEnv implementation (CUDA, CyberBattle, Hybrid).
 * 
 * Features:
 * - Experience replay buffer
 * - Epsilon-greedy exploration
 * - GPU-accelerated Q-table updates
 ************************************************************/

#ifndef TABULAR_QLEARNING_HPP
#define TABULAR_QLEARNING_HPP

#include "IEnv.hpp"
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <random>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace cyber_rl {

/////////////////////////////////////////////////////////////
// CONFIGURATION
/////////////////////////////////////////////////////////////

struct QLearningConfig {
    // Learning parameters
    float learning_rate = 0.01f;     // Alpha
    float discount_factor = 0.99f;   // Gamma
    float epsilon = 0.1f;            // Exploration rate
    
    // Replay buffer
    int replay_capacity = 100000;
    int min_replay_size = 1000;      // Start training after this many samples
    int batch_size = 256;
    
    // Training
    int max_episodes = 5000;
    int eval_frequency = 300;        // Evaluate every N episodes
    int num_eval_episodes = 10;      // Episodes per evaluation
    
    // Multi-threading
    int num_workers = 4;             // Worker threads for experience collection
};


/////////////////////////////////////////////////////////////
// EXPERIENCE REPLAY
/////////////////////////////////////////////////////////////

struct Transition {
    int state;          // Discretized state (position)
    int action;         // Action taken
    float reward;       // Immediate reward
    int next_state;     // Resulting state
    bool done;          // Episode terminated
    
    Transition() : state(0), action(0), reward(0.0f), next_state(0), done(false) {}
};

class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacity);
    
    // Add transition to buffer
    void push(const Transition& t);
    
    // Sample random batch
    std::vector<Transition> sample(size_t batch_size, std::mt19937& rng);
    
    // Get current size
    size_t size() const { return size_.load(); }
    
    // Check if ready for training
    bool ready_for_training(size_t min_size) const { 
        return size() >= min_size; 
    }
    
private:
    std::vector<Transition> buffer_;
    std::atomic<size_t> write_pos_;
    std::atomic<size_t> size_;
};


/////////////////////////////////////////////////////////////
// TRAINING METRICS
/////////////////////////////////////////////////////////////

struct TrainingMetrics {
    std::atomic<int> episodes_completed{0};
    std::atomic<int> successful_episodes{0};
    std::atomic<long long> total_steps{0};
    std::atomic<float> avg_episode_reward{0.0f};
    
    void reset() {
        episodes_completed = 0;
        successful_episodes = 0;
        total_steps = 0;
        avg_episode_reward = 0.0f;
    }
    
    float success_rate() const {
        int eps = episodes_completed.load();
        return eps > 0 ? (100.0f * successful_episodes.load() / eps) : 0.0f;
    }
};


/////////////////////////////////////////////////////////////
// TABULAR Q-LEARNING AGENT
/////////////////////////////////////////////////////////////

class TabularQLearning {
public:
    TabularQLearning(IEnv* env, const QLearningConfig& config = QLearningConfig());
    ~TabularQLearning();
    
    // Disable copy
    TabularQLearning(const TabularQLearning&) = delete;
    TabularQLearning& operator=(const TabularQLearning&) = delete;
    
    
    //--------------------------------------------------------
    // Training
    //--------------------------------------------------------
    
    // Train agent for specified number of episodes
    void train(int num_episodes);
    
    // Run single episode (for worker threads)
    float run_episode(int worker_id);
    
    
    //--------------------------------------------------------
    // Evaluation
    //--------------------------------------------------------
    
    // Evaluate policy (greedy, no exploration)
    float evaluate(int num_episodes = 10);
    
    
    //--------------------------------------------------------
    // Policy Queries
    //--------------------------------------------------------
    
    // Get best action for state (greedy)
    int get_best_action(int state) const;
    
    // Get action with epsilon-greedy exploration
    int get_action(int state, float epsilon, std::mt19937& rng);
    
    // Get Q-value for state-action pair
    float get_q_value(int state, int action) const;
    
    
    //--------------------------------------------------------
    // Metrics
    //--------------------------------------------------------
    
    const TrainingMetrics& get_metrics() const { return metrics_; }
    
    void print_metrics() const;
    
    
private:
    //--------------------------------------------------------
    // Q-Table Management
    //--------------------------------------------------------
    
    void initialize_q_table();
    void update_q_table_gpu(const std::vector<Transition>& batch);
    void sync_q_table_to_host();
    
    
    //--------------------------------------------------------
    // Worker Thread
    //--------------------------------------------------------
    
    void worker_thread(int worker_id);
    
    
    //--------------------------------------------------------
    // Member Variables
    //--------------------------------------------------------
    
    IEnv* env_;
    QLearningConfig config_;
    
    // Q-table (state_dim × action_dim)
    float* d_q_table_;              // Device (GPU)
    std::vector<float> h_q_table_;  // Host (CPU) - for workers
    int num_states_;
    int num_actions_;
    mutable std::mutex q_table_mutex_;      // Protect host Q-table access
    
    // Experience replay
    std::unique_ptr<ReplayBuffer> replay_buffer_;
    
    // Training state
    TrainingMetrics metrics_;
    std::atomic<bool> stop_training_;
    
    // CUDA resources
    curandState* d_rng_states_;
};

} // namespace cyber_rl

#endif // TABULAR_QLEARNING_HPP


/************************************************************
 * DESIGN NOTES
 * 
 * Q-Table Structure:
 * - Tabular: Q[state][action] = expected return
 * - State = agent position (0 to 50,000 for CUDA env)
 * - Actions = 11 (8 moves + 3 defense actions)
 * - Memory: 50k × 11 × 4 bytes = 2.2 MB
 * 
 * Training Flow:
 * 1. Worker threads collect experience → replay buffer
 * 2. Main thread samples batches → GPU Q-table update
 * 3. Periodically sync Q-table GPU → CPU for workers
 * 
 * Epsilon-Greedy:
 * - With probability ε: random action (explore)
 * - With probability 1-ε: best Q-value action (exploit)
 * 
 * Q-Learning Update:
 * Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
 * 
 * GPU Acceleration:
 * - Q-table updates batched on GPU (256 transitions)
 * - ~10x faster than CPU for large state spaces
 ************************************************************/