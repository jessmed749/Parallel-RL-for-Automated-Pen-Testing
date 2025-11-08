#include "worker_thread.h"
#include <random>
#include <iostream>

// CPUPolicy implementation
CPUPolicy::CPUPolicy(unsigned int seed) {
    // constructor intentionally does not rely on class members for RNG or epsilon,
    // select_action uses a thread-local RNG and a local epsilon.
    (void)seed;
}

int CPUPolicy::select_action(const std::array<int, 5>& obs, int action_space_size) {
    // Use a thread-local RNG 
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float epsilon = 0.1f;
    
    // Epsilon-greedy: explore with probability epsilon
    if (dist(rng) < epsilon) {
        // Random action
        std::uniform_int_distribution<int> action_dist(0, action_space_size - 1);
        return action_dist(rng);
    }
    
    // Greedy heuristic: prefer EXPLOIT_NEXT (action 1)
    // Simple strategy: always try to move forward
    return EXPLOIT_NEXT;
}


// WorkerThread implementation
WorkerThread::WorkerThread(int id,
                           std::shared_ptr<RequestQueue> req_queue,
                           std::shared_ptr<ResponseQueue> resp_queue,
                           unsigned int seed,
                           int max_episodes)
    : id_(id),
      req_queue_(req_queue),
      resp_queue_(resp_queue),
      env_(seed + id),  // Unique seed per thread
      cpu_policy_(seed + id + 10000),
      max_episodes_(max_episodes),
      running_(true),
      request_id_counter_(0),
      episodes_completed_(0),
      total_reward_(0.0f),
      successes_(0),
      gpu_timeouts_(0) {}

void WorkerThread::run() {
    for (int episode = 0; episode < max_episodes_ && running_.load(); ++episode) {
        // Reset environment for new episode
        env_.reset();
        float episode_reward = 0.0f;
        bool done = false;
        int steps = 0;
        const int MAX_STEPS = 100;  // Prevent infinite loops
        
        // Run episode
        while (!done && running_.load() && steps < MAX_STEPS) {
            // Get current observation
            auto obs = env_.get_observation();
            
            // Convert observation to State for queue
            State state;
            for (int i = 0; i < 5; i++) {
                state[i] = static_cast<float>(obs[i]);
            }
            
            // Request action from GPU (with CPU fallback)
            int action = request_action_from_gpu(state, 10);  // 10ms timeout
            
            // Execute action in environment
            int reward = env_.step(static_cast<Action>(action));
            
            // Update episode reward
            episode_reward += reward;
            
            // Check if episode is done
            done = env_.is_episode_done();
            
            steps++;
        }
        
        // Update statistics
        episodes_completed_.fetch_add(1);
        
        // Update total reward (atomic float operations)
        float current_total = total_reward_.load();
        while (!total_reward_.compare_exchange_weak(current_total, 
                                                     current_total + episode_reward)) {
            // Retry if another thread modified it
        }
        
        // Check if episode was successful (completed with bonus)
        if (episode_reward >= 50.0f) {
            successes_.fetch_add(1);
        }
    }
}

void WorkerThread::stop() {
    running_.store(false);
}

int WorkerThread::request_action_from_gpu(const State& state, int timeout_ms) {
    // Generate unique request ID
    int req_id = generate_request_id();
    
    // Create and push state request
    StateRequest req(req_id, id_, state);
    
    try {
        req_queue_->push(req);
    } catch (...) {
        // Queue push failed, use CPU fallback
        gpu_timeouts_.fetch_add(1);
        return select_action_cpu_fallback();
    }
    
    // Wait for response
    ActionResponse resp;
    bool got_response = resp_queue_->try_pop(req_id, resp, timeout_ms);
    
    if (got_response) {
        return resp.action;
    } else {
        // Timeout: use CPU fallback
        gpu_timeouts_.fetch_add(1);
        return select_action_cpu_fallback();
    }
}

int WorkerThread::select_action_cpu_fallback() {
    // Get current observation
    auto obs = env_.get_observation();
    
    // Use CPU policy to select action
    return cpu_policy_.select_action(obs, env_.get_action_space_size());
}
