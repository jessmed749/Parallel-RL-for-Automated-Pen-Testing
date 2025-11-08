#ifndef WORKER_THREAD_H
#define WORKER_THREAD_H

#include "../../../include/data_structures.h"
#include "../environment/pentest_env.h"
#include "../queues/request_queue.h"
#include "../queues/response_queue.h"
#include <memory>
#include <atomic>
#include <random>

// simple fall back cpu policy
class CPUPolicy {
public:
        explicit CPUPolicy(unsigned int seed);

    // select action based on epsilon greedy
    // Returns action index (0-3)
    int select_action(const std::array<int, 5>& obs, int current_action_space);

private:
    std::mt19937 rng;
    float epsilon_; // exploration rate, 0.1
};

// worker thread class
class WorkerThread {
public: 
    WorkerThread (int id, std::shared_ptr<RequestQueue> req_queue, std::shared_ptr<ResponseQueue> resp_queue, unsigned int seed, int max_episodes);

    // destructor
    ~WorkerThread() = default;

    // main sim loop
    void run();

    // signal thread to stop
    void stop();

    // get stats
    int get_episodes_completed() const { return episodes_completed_.load(); }
    float get_total_reward() const { return total_reward_.load(); }
    int get_successes() const { return successes_.load(); }
    int get_gpu_timeouts() const { return gpu_timeouts_.load(); }

    // compute stats
    float get_average_reward() const {
        int eps = episodes_completed_.load();
        return (eps > 0) ? (total_reward_.load() / eps) : 0.0f;
    }
    
    float get_success_rate() const {
        int eps = episodes_completed_.load();
        return (eps > 0) ? (100.0f * successes_.load() / eps) : 0.0f;
    }

private:
    // thread id
    int id_;

    //shared queues
    std::shared_ptr<RequestQueue> req_queue_;
    std::shared_ptr<ResponseQueue> resp_queue_;

    //local instances
    PentestEnv env_;
    CPUPolicy cpu_policy_;

    //configs
    int max_episodes_;
    std::atomic<bool> running_;

    //requst id generation
    int request_id_counter_;
    int generate_request_id(){
        return id_ * 1000000 + (request_id_counter_++);
    }

    //stats, atomic for thread safety
    std::atomic<int> episodes_completed_;
    std::atomic<float> total_reward_;
    std::atomic<int> successes_;
    std::atomic<int> gpu_timeouts_;

    // helper functions/methods
    int request_action_from_gpu(const State& state, int timeout_ms);
    int select_action_cpu_fallback();
};

#endif