#include "worker_thread.h"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <cassert>

// ANSI color codes
#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define RESET "\033[0m"

void print_test(const std::string& name, bool passed){
    std::cout << (passed ? GREEN "✓ " : RED "✗ ") 
              << name << RESET << std::endl;
}

void print_header(const std::string& name) {
    std::cout << "\n" << BLUE << "=== " << name << " ===" << RESET << std::endl;
}

// Test 1: Single worker with CPU fallback only
void test_single_worker_cpu_only() {
    print_header("Test 1: Single Worker (CPU Fallback Only)");
    
    auto req_queue = std::make_shared<RequestQueue>(100);
    auto resp_queue = std::make_shared<ResponseQueue>();
    
    WorkerThread worker(0, req_queue, resp_queue, 42, 10);
    
    // Run worker (no GPU thread, so it will use CPU fallback)
    std::thread t([&]() { worker.run(); });
    t.join();
    
    print_test("Episodes completed", worker.get_episodes_completed() == 10);
    print_test("Has timeouts (CPU fallback used)", worker.get_gpu_timeouts() > 0);
    print_test("Total reward tracked", worker.get_total_reward() != 0.0f);
    
    std::cout << "  Episodes: " << worker.get_episodes_completed() << std::endl;
    std::cout << "  Avg reward: " << worker.get_average_reward() << std::endl;
    std::cout << "  Success rate: " << worker.get_success_rate() << "%" << std::endl;
    std::cout << "  GPU timeouts: " << worker.get_gpu_timeouts() << std::endl;
}

// Test 2: Single worker with simulated GPU
void test_single_worker_with_gpu() {
    print_header("Test 2: Single Worker with Simulated GPU");
    
    auto req_queue = std::make_shared<RequestQueue>(100);
    auto resp_queue = std::make_shared<ResponseQueue>();
    
    WorkerThread worker(0, req_queue, resp_queue, 42, 10);
    
    // Simulate GPU thread responding immediately
    std::atomic<bool> gpu_running(true);
    std::thread gpu_thread([&]() {
        while (gpu_running.load()) {
            auto batch = req_queue->try_pop_batch(8, 10);
            for (const auto& req : batch) {
                // Simple policy: always try EXPLOIT_NEXT
                ActionResponse resp(req.request_id, EXPLOIT_NEXT, 0.9f);
                resp_queue->push(req.request_id, resp);
            }
        }
    });
    
    // Run worker
    std::thread worker_thread([&]() { worker.run(); });
    worker_thread.join();
    
    gpu_running.store(false);
    gpu_thread.join();
    
    print_test("Episodes completed", worker.get_episodes_completed() == 10);
    print_test("GPU responses received", worker.get_gpu_timeouts() < worker.get_episodes_completed() * 10);
    
    std::cout << "  Episodes: " << worker.get_episodes_completed() << std::endl;
    std::cout << "  Avg reward: " << worker.get_average_reward() << std::endl;
    std::cout << "  Success rate: " << worker.get_success_rate() << "%" << std::endl;
    std::cout << "  GPU timeouts: " << worker.get_gpu_timeouts() << std::endl;
}

// Test 3: Multiple workers in parallel
void test_multiple_workers() {
    print_header("Test 3: Multiple Workers in Parallel");
    
    const int NUM_WORKERS = 4;
    const int EPISODES_PER_WORKER = 20;
    
    auto req_queue = std::make_shared<RequestQueue>(1000);
    auto resp_queue = std::make_shared<ResponseQueue>();
    
    std::vector<std::unique_ptr<WorkerThread>> workers;
    std::vector<std::thread> worker_threads;
    
    // Create workers
    for (int i = 0; i < NUM_WORKERS; i++) {
        workers.push_back(std::make_unique<WorkerThread>(
            i, req_queue, resp_queue, 42 + i, EPISODES_PER_WORKER
        ));
    }
    
    // Simulate GPU thread
    std::atomic<bool> gpu_running(true);
    std::thread gpu_thread([&]() {
        int processed = 0;
        while (gpu_running.load() || !req_queue->empty()) {
            auto batch = req_queue->try_pop_batch(16, 10);
            for (const auto& req : batch) {
                // Random action
                int action = processed % 4;
                ActionResponse resp(req.request_id, action, 0.8f);
                resp_queue->push(req.request_id, resp);
                processed++;
            }
        }
    });
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Start workers
    for (auto& worker : workers) {
        worker_threads.emplace_back([&worker]() { worker->run(); });
    }
    
    // Wait for all workers
    for (auto& thread : worker_threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    // Stop GPU thread
    gpu_running.store(false);
    gpu_thread.join();
    
    // Collect statistics
    int total_episodes = 0;
    float total_reward = 0.0f;
    int total_successes = 0;
    int total_timeouts = 0;
    
    std::cout << "\nPer-Worker Statistics:" << std::endl;
    for (int i = 0; i < NUM_WORKERS; i++) {
        int eps = workers[i]->get_episodes_completed();
        float reward = workers[i]->get_total_reward();
        int succ = workers[i]->get_successes();
        int timeouts = workers[i]->get_gpu_timeouts();
        
        std::cout << "  Worker " << i << ": "
                  << eps << " episodes, "
                  << "reward=" << reward << ", "
                  << "successes=" << succ << ", "
                  << "timeouts=" << timeouts << std::endl;
        
        total_episodes += eps;
        total_reward += reward;
        total_successes += succ;
        total_timeouts += timeouts;
    }
    
    print_test("All episodes completed", total_episodes == NUM_WORKERS * EPISODES_PER_WORKER);
    
    std::cout << "\nAggregate Statistics:" << std::endl;
    std::cout << "  Total episodes: " << total_episodes << std::endl;
    std::cout << "  Total reward: " << total_reward << std::endl;
    std::cout << "  Avg reward: " << (total_reward / total_episodes) << std::endl;
    std::cout << "  Success rate: " << (100.0f * total_successes / total_episodes) << "%" << std::endl;
    std::cout << "  Timeout rate: " << (100.0f * total_timeouts / total_episodes) << "%" << std::endl;
    std::cout << "  Wall-clock time: " << duration << " ms" << std::endl;
    std::cout << "  Throughput: " << (1000.0f * total_episodes / duration) << " episodes/sec" << std::endl;
}

// Test 4: Worker stop functionality
void test_worker_stop() {
    print_header("Test 4: Worker Stop Functionality");
    
    auto req_queue = std::make_shared<RequestQueue>(100);
    auto resp_queue = std::make_shared<ResponseQueue>();
    
    WorkerThread worker(0, req_queue, resp_queue, 42, 1000);  // Many episodes
    
    // Start worker
    std::thread t([&]() { worker.run(); });
    
    // Let it run for a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Stop worker
    worker.stop();
    t.join();
    
    int completed = worker.get_episodes_completed();
    print_test("Worker stopped before completion", completed < 1000);
    print_test("Worker completed some episodes", completed > 0);
    
    std::cout << "  Episodes completed before stop: " << completed << std::endl;
}

// Test 5: Stress test with many workers
void test_stress() {
    print_header("Test 5: Stress Test (16 workers, 50 episodes each)");
    
    const int NUM_WORKERS = 16;
    const int EPISODES_PER_WORKER = 50;
    
    auto req_queue = std::make_shared<RequestQueue>(5000);
    auto resp_queue = std::make_shared<ResponseQueue>();
    
    std::vector<std::unique_ptr<WorkerThread>> workers;
    std::vector<std::thread> worker_threads;
    
    // Create workers
    for (int i = 0; i < NUM_WORKERS; i++) {
        workers.push_back(std::make_unique<WorkerThread>(
            i, req_queue, resp_queue, 100 + i, EPISODES_PER_WORKER
        ));
    }
    
    // Fast GPU simulator
    std::atomic<bool> gpu_running(true);
    std::thread gpu_thread([&]() {
        while (gpu_running.load() || !req_queue->empty()) {
            auto batch = req_queue->try_pop_batch(64, 5);
            for (const auto& req : batch) {
                ActionResponse resp(req.request_id, EXPLOIT_NEXT, 1.0f);
                resp_queue->push(req.request_id, resp);
            }
        }
    });
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Start workers
    for (auto& worker : workers) {
        worker_threads.emplace_back([&worker]() { worker->run(); });
    }
    
    // Wait for completion
    for (auto& thread : worker_threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    gpu_running.store(false);
    gpu_thread.join();
    
    // Aggregate results
    int total_episodes = 0;
    for (const auto& worker : workers) {
        total_episodes += worker->get_episodes_completed();
    }
    
    print_test("All episodes completed", total_episodes == NUM_WORKERS * EPISODES_PER_WORKER);
    
    std::cout << YELLOW << "  Total episodes: " << total_episodes << std::endl;
    std::cout << "  Duration: " << duration << " ms" << std::endl;
    std::cout << "  Throughput: " << (1000.0f * total_episodes / duration) 
              << " episodes/sec" << RESET << std::endl;
}

int main() {
    std::cout << "\n" << BLUE << "╔═══════════════════════════════════════╗" << RESET << std::endl;
    std::cout << BLUE << "║   Worker Thread Test Suite            ║" << RESET << std::endl;
    std::cout << BLUE << "╚═══════════════════════════════════════╝" << RESET << "\n" << std::endl;
    
    try {
        test_single_worker_cpu_only();
        test_single_worker_with_gpu();
        test_multiple_workers();
        test_worker_stop();
        test_stress();
        
        std::cout << "\n" << GREEN << "All worker thread tests passed! ✓" << RESET << "\n" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "\n" << RED << "Test failed with exception: " 
                  << e.what() << RESET << std::endl;
        return 1;
    }
}