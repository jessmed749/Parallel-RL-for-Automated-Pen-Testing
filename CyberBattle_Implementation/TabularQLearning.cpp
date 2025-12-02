/************************************************************
 * FILE 8: TabularQLearning.cpp
 * 
 * Implementation of tabular Q-learning with GPU acceleration.
 ************************************************************/

#include "TabularQLearning.hpp"
#include <iostream>
#include <algorithm>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
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

extern "C" {
    void launch_q_learning_update(
        const int* d_states,
        const int* d_actions,
        const float* d_rewards,
        const int* d_next_states,
        const bool* d_dones,
        float* d_q_table,
        int batch_size,
        int num_actions,
        float alpha,
        float gamma
    );
}


/////////////////////////////////////////////////////////////
// ReplayBuffer Implementation
/////////////////////////////////////////////////////////////

ReplayBuffer::ReplayBuffer(size_t capacity)
    : buffer_(capacity)
    , write_pos_(0)
    , size_(0)
{}

void ReplayBuffer::push(const Transition& t) {
    size_t pos = write_pos_.fetch_add(1) % buffer_.size();
    buffer_[pos] = t;
    size_.store(std::min(buffer_.size(), size_.load() + 1));
}

std::vector<Transition> ReplayBuffer::sample(size_t batch_size, std::mt19937& rng) {
    std::vector<Transition> batch;
    size_t n = std::min(batch_size, size());
    std::uniform_int_distribution<size_t> dist(0, size() - 1);
    
    batch.reserve(n);
    for (size_t i = 0; i < n; i++) {
        batch.push_back(buffer_[dist(rng)]);
    }
    
    return batch;
}


/////////////////////////////////////////////////////////////
// TabularQLearning Implementation
/////////////////////////////////////////////////////////////

TabularQLearning::TabularQLearning(IEnv* env, const QLearningConfig& config)
    : env_(env)
    , config_(config)
    , d_q_table_(nullptr)
    , d_rng_states_(nullptr)
    , stop_training_(false)
{
    num_states_ = 50000;  // Fixed for now (CUDA env size)
    num_actions_ = env_->get_action_dim();
    
    // Initialize Q-table
    initialize_q_table();
    
    // Create replay buffer
    replay_buffer_ = std::make_unique<ReplayBuffer>(config_.replay_capacity);
    
    std::cout << "[Q-Learning] Initialized with " << num_states_ 
              << " states, " << num_actions_ << " actions\n";
}

TabularQLearning::~TabularQLearning() {
    if (d_q_table_) {
        cudaFree(d_q_table_);
    }
    if (d_rng_states_) {
        cudaFree(d_rng_states_);
    }
}

void TabularQLearning::initialize_q_table() {
    // Allocate host Q-table
    h_q_table_.resize(num_states_ * num_actions_, 0.0f);
    
    // Allocate device Q-table
    size_t q_table_size = num_states_ * num_actions_ * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_q_table_, q_table_size));
    CUDA_CHECK(cudaMemset(d_q_table_, 0, q_table_size));
    
    std::cout << "[Q-Learning] Q-table allocated: " 
              << (q_table_size / 1024.0 / 1024.0) << " MB\n";
}

void TabularQLearning::train(int num_episodes) {
    std::cout << "\n=== TRAINING START ===\n";
    std::cout << "Episodes: " << num_episodes << "\n";
    std::cout << "Workers: " << config_.num_workers << "\n\n";
    
    metrics_.reset();
    stop_training_ = false;
    
    // Launch worker threads
    std::vector<std::thread> workers;
    for (int i = 0; i < config_.num_workers; i++) {
        workers.emplace_back(&TabularQLearning::worker_thread, this, i);
    }
    
    // Training loop
    auto start_time = std::chrono::steady_clock::now();
    std::mt19937 sample_rng(42);
    
    for (int episode = 0; episode < num_episodes; episode++) {
        
        // Wait for minimum replay size
        while (!replay_buffer_->ready_for_training(config_.min_replay_size)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Sample batch and update Q-table
        auto batch = replay_buffer_->sample(config_.batch_size, sample_rng);
        update_q_table_gpu(batch);
        
        // Periodically sync Q-table to host for workers
        if (episode % 10 == 0) {
            sync_q_table_to_host();
        }
        
        // Evaluation
        if (episode % config_.eval_frequency == 0) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            
            std::cout << "Episode " << episode << " [" << seconds << "s]\n";
            print_metrics();
            
            // Run evaluation episodes
            float eval_reward = evaluate(config_.num_eval_episodes);
            std::cout << "  Eval reward: " << eval_reward << "\n\n";
        }
    }
    
    // Stop workers
    stop_training_ = true;
    for (auto& t : workers) {
        t.join();
    }
    
    std::cout << "\n=== TRAINING COMPLETE ===\n";
    print_metrics();
}

void TabularQLearning::worker_thread(int worker_id) {
    std::mt19937 rng(worker_id + std::chrono::steady_clock::now().time_since_epoch().count());
    
    while (!stop_training_.load()) {
        float episode_reward = run_episode(worker_id);
        
        // Update metrics
        metrics_.episodes_completed++;
        
        // Update running average reward
        float old_avg = metrics_.avg_episode_reward.load();
        float new_avg = old_avg + (episode_reward - old_avg) / metrics_.episodes_completed.load();
        metrics_.avg_episode_reward.store(new_avg);
    }
}

float TabularQLearning::run_episode(int worker_id) {
    std::mt19937 rng(worker_id + std::chrono::steady_clock::now().time_since_epoch().count());
    
    State state = env_->reset();
    float total_reward = 0.0f;
    int steps = 0;
    
    while (steps < 500) {  // Max episode length
        // Get action with epsilon-greedy
        int action = get_action(state.position, config_.epsilon, rng);
        
        // Take step
        StepResult result = env_->step(action);
        
        // Store transition
        Transition t;
        t.state = state.position;
        t.action = action;
        t.reward = result.reward;
        t.next_state = result.next_state.position;
        t.done = result.done;
        
        replay_buffer_->push(t);
        
        total_reward += result.reward;
        steps++;
        metrics_.total_steps++;
        
        if (result.done) {
            // SUCCESS CRITERION CHANGED:
            // Consider an episode successful if the total return is
            // not too negative. With the new shaped rewards, genuinely
            // good behavior should have higher (less negative or positive)
            // returns than catastrophic outbreaks.
            if (total_reward > -10.0f) {
                metrics_.successful_episodes++;
            }
            break;
        }
        
        state = result.next_state;
    }
    
    return total_reward;
}

float TabularQLearning::evaluate(int num_episodes) {
    float total_reward = 0.0f;
    
    for (int ep = 0; ep < num_episodes; ep++) {
        State state = env_->reset();
        float episode_reward = 0.0f;
        int steps = 0;
        
        while (steps < 500) {
            // Greedy action (no exploration)
            int action = get_best_action(state.position);
            
            StepResult result = env_->step(action);
            episode_reward += result.reward;
            steps++;
            
            if (result.done) break;
            
            state = result.next_state;
        }
        
        total_reward += episode_reward;
    }
    
    return total_reward / num_episodes;
}

void TabularQLearning::update_q_table_gpu(const std::vector<Transition>& batch) {
    if (batch.empty()) return;
    
    // Prepare host arrays
    std::vector<int> h_states(batch.size());
    std::vector<int> h_actions(batch.size());
    std::vector<float> h_rewards(batch.size());
    std::vector<int> h_next_states(batch.size());
    std::vector<bool> h_dones(batch.size());  // Keep as bool
    
    for (size_t i = 0; i < batch.size(); i++) {
        h_states[i]   = batch[i].state;
        h_actions[i]  = batch[i].action;
        h_rewards[i]  = batch[i].reward;
        h_next_states[i] = batch[i].next_state;
        h_dones[i]    = batch[i].done;
    }
    
    // Allocate device memory
    int*   d_states;
    int*   d_actions;
    float* d_rewards;
    int*   d_next_states;
    bool*  d_dones;
    
    CUDA_CHECK(cudaMalloc(&d_states,      batch.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_actions,     batch.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rewards,     batch.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next_states, batch.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dones,       batch.size() * sizeof(bool)));
    
    // Copy to device - copy bool array directly as bytes
    CUDA_CHECK(cudaMemcpy(d_states,      h_states.data(), 
                          batch.size() * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_actions,     h_actions.data(),
                          batch.size() * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rewards,     h_rewards.data(),
                          batch.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_next_states, h_next_states.data(),
                          batch.size() * sizeof(int),   cudaMemcpyHostToDevice));
    
    // Convert bool vector to byte array for copying
    std::vector<unsigned char> h_dones_bytes(batch.size());
    for (size_t i = 0; i < batch.size(); i++) {
        h_dones_bytes[i] = h_dones[i] ? 1 : 0;
    }
    CUDA_CHECK(cudaMemcpy(d_dones, h_dones_bytes.data(),
                          batch.size() * sizeof(bool), cudaMemcpyHostToDevice));
    
    // Launch kernel
    launch_q_learning_update(
        d_states,
        d_actions,
        d_rewards,
        d_next_states,
        d_dones,
        d_q_table_,
        static_cast<int>(batch.size()),
        num_actions_,
        config_.learning_rate,
        config_.discount_factor
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Free device memory
    cudaFree(d_states);
    cudaFree(d_actions);
    cudaFree(d_rewards);
    cudaFree(d_next_states);
    cudaFree(d_dones);
}

void TabularQLearning::sync_q_table_to_host() {
    std::lock_guard<std::mutex> lock(q_table_mutex_);
    
    CUDA_CHECK(cudaMemcpy(h_q_table_.data(), d_q_table_,
                          h_q_table_.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

int TabularQLearning::get_best_action(int state) const {
    std::lock_guard<std::mutex> lock(q_table_mutex_);
    
    int best_action = 0;
    float best_q = -1e9f;
    
    for (int a = 0; a < num_actions_; a++) {
        float q = h_q_table_[state * num_actions_ + a];
        if (q > best_q) {
            best_q = q;
            best_action = a;
        }
    }
    
    return best_action;
}

int TabularQLearning::get_action(int state, float epsilon, std::mt19937& rng) {
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    
    if (uni(rng) < epsilon) {
        // Explore: random action
        std::uniform_int_distribution<int> act_dist(0, num_actions_ - 1);
        return act_dist(rng);
    } else {
        // Exploit: best action
        return get_best_action(state);
    }
}

float TabularQLearning::get_q_value(int state, int action) const {
    std::lock_guard<std::mutex> lock(q_table_mutex_);
    return h_q_table_[state * num_actions_ + action];
}

void TabularQLearning::print_metrics() const {
    int episodes = metrics_.episodes_completed.load();
    int successes = metrics_.successful_episodes.load();
    long long steps = metrics_.total_steps.load();
    float avg_reward = metrics_.avg_episode_reward.load();
    float success_rate = metrics_.success_rate();
    
    std::cout << "  Episodes: " << episodes 
              << " | Successes: " << successes
              << " | Success Rate: " << success_rate << "%"
              << " | Total Steps: " << steps
              << " | Avg Reward: " << avg_reward << "\n";
}

} // namespace cyber_rl