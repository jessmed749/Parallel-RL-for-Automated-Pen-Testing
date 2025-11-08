#include "request_queue.h"
#include "response_queue.h"
#include <iostream>
#include <thread>
#include <vector>
#include <cassert>
#include <atomic>
#include <chrono>

// ANSI color codes for pretty output
#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define RESET "\033[0m"

void print_test(const std::string& name, bool passed) {
    std::cout << (passed ? GREEN "✓ " : RED "✗ ") 
              << name << RESET << std::endl;
}

void print_header(const std::string& name) {
    std::cout << "\n" << BLUE << "=== " << name << " ===" << RESET << std::endl;
}

// Test 1: Basic RequestQueue push/pop
void test_request_queue_basic() {
    print_header("Test 1: RequestQueue Basic Operations");
    
    RequestQueue queue;
    StateRequest req1(1, 0, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    
    queue.push(req1);
    print_test("Push single request", queue.size() == 1);
    
    StateRequest out;
    bool success = queue.try_pop(out, 100);
    print_test("Pop single request", success && out.request_id == 1);
    print_test("Queue empty after pop", queue.empty());
}

// Test 2: RequestQueue timeout behavior
void test_request_queue_timeout() {
    print_header("Test 2: RequestQueue Timeout Behavior");
    
    RequestQueue queue;
    StateRequest req;
    
    auto start = std::chrono::steady_clock::now();
    bool success = queue.try_pop(req, 100);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    print_test("Timeout returns false", !success);
    print_test("Timeout duration correct", 
               duration.count() >= 100 && duration.count() < 150);
}

// Test 3: RequestQueue concurrent push
void test_request_queue_concurrent_push() {
    print_header("Test 3: RequestQueue Concurrent Push");
    
    RequestQueue queue;
    const int num_threads = 10;
    const int items_per_thread = 100;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&queue, i, items_per_thread]() {
            for (int j = 0; j < items_per_thread; j++) {
                StateRequest req(i * 1000 + j, i, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
                queue.push(req);
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    print_test("All items pushed", queue.size() == num_threads * items_per_thread);
}

// Test 4: RequestQueue batch operations
void test_request_queue_batch() {
    print_header("Test 4: RequestQueue Batch Operations");
    
    RequestQueue queue;
    
    // Push 100 requests
    for (int i = 0; i < 100; i++) {
        queue.push(StateRequest(i, 0, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    }
    
    // Pop batch of 64
    auto batch = queue.try_pop_batch(64, 100);
    
    print_test("Batch size correct", batch.size() == 64);
    print_test("Remaining items correct", queue.size() == 36);
    
    // Verify batch contains correct requests (first 64)
    bool ids_correct = true;
    for (size_t i = 0; i < batch.size(); i++) {
        if (batch[i].request_id != static_cast<int>(i)) {
            ids_correct = false;
            break;
        }
    }
    print_test("Batch IDs in order", ids_correct);
}

// Test 5: RequestQueue capacity enforcement
void test_request_queue_capacity() {
    print_header("Test 5: RequestQueue Capacity Enforcement");
    
    RequestQueue queue(10);  // Small capacity for testing
    
    // Fill the queue
    for (int i = 0; i < 10; i++) {
        queue.push(StateRequest(i, 0, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    }
    
    print_test("Queue at max capacity", queue.size() == 10);
    
    // Try to push one more (should block unless we pop)
    std::atomic<bool> push_completed(false);
    std::thread pusher([&]() {
        queue.push(StateRequest(10, 0, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
        push_completed = true;
    });
    
    // Wait a bit - push should be blocked
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    print_test("Push blocks when full", !push_completed);
    
    // Pop one item - this should unblock the pusher
    StateRequest dummy;
    queue.try_pop(dummy, 100);
    
    pusher.join();
    print_test("Push completes after pop", push_completed);
}

// Test 6: ResponseQueue basic operations
void test_response_queue_basic() {
    print_header("Test 6: ResponseQueue Basic Operations");
    
    ResponseQueue queue;
    ActionResponse resp1(1, 2, 0.95f);  // request_id=1, action=2
    
    queue.push(1, resp1);
    print_test("Response pushed", queue.has_response(1));
    
    ActionResponse out;
    bool success = queue.try_pop(1, out, 100);
    
    print_test("Response popped", success);
    print_test("Response action correct", out.action == 2);
    print_test("Response removed after pop", !queue.has_response(1));
}

// Test 7: ResponseQueue timeout
void test_response_queue_timeout() {
    print_header("Test 7: ResponseQueue Timeout");
    
    ResponseQueue queue;
    ActionResponse resp;
    
    auto start = std::chrono::steady_clock::now();
    bool success = queue.try_pop(999, resp, 100);  // Non-existent ID
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    print_test("Timeout on missing response", !success);
    print_test("Timeout duration correct", 
               duration.count() >= 100 && duration.count() < 150);
}

// Test 8: ResponseQueue concurrent access
void test_response_queue_concurrent() {
    print_header("Test 8: ResponseQueue Concurrent Access");
    
    ResponseQueue queue;
    const int num_workers = 8;
    std::vector<std::thread> workers;
    std::atomic<int> success_count(0);
    
    // GPU thread pushes responses
    std::thread gpu_thread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        for (int i = 0; i < num_workers; i++) {
            queue.push(i, ActionResponse(i, i % 4, 1.0f));
        }
    });
    
    // Worker threads wait for their responses
    for (int i = 0; i < num_workers; i++) {
        workers.emplace_back([&queue, i, &success_count]() {
            ActionResponse resp;
            bool success = queue.try_pop(i, resp, 200);
            if (success && resp.request_id == i) {
                success_count++;
            }
        });
    }
    
    gpu_thread.join();
    for (auto& w : workers) {
        w.join();
    }
    
    print_test("All workers got responses", success_count == num_workers);
}

// Test 9: End-to-end producer-consumer
void test_end_to_end_flow() {
    print_header("Test 9: End-to-End Producer-Consumer");
    
    RequestQueue req_queue;
    ResponseQueue resp_queue;
    std::atomic<int> episodes_completed(0);
    const int num_workers = 4;
    const int episodes_per_worker = 10;
    
    // GPU thread: processes batches
    std::thread gpu_thread([&]() {
        while (episodes_completed < num_workers * episodes_per_worker) {
            auto batch = req_queue.try_pop_batch(8, 10);
            for (const auto& req : batch) {
                // Simulate GPU inference
                ActionResponse resp(req.request_id, req.request_id % 4, 0.9f);
                resp_queue.push(req.request_id, resp);
            }
        }
    });
    
    // Worker threads: run episodes
    std::vector<std::thread> workers;
    for (int worker_id = 0; worker_id < num_workers; worker_id++) {
        workers.emplace_back([&, worker_id]() {
            for (int ep = 0; ep < episodes_per_worker; ep++) {
                int req_id = worker_id * 1000000 + ep;
                StateRequest req(req_id, worker_id, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
                
                req_queue.push(req);
                
                ActionResponse resp;
                bool success = resp_queue.try_pop(req_id, resp, 100);
                
                if (success) {
                    episodes_completed++;
                }
            }
        });
    }
    
    for (auto& w : workers) {
        w.join();
    }
    gpu_thread.join();
    
    print_test("All episodes completed", 
               episodes_completed == num_workers * episodes_per_worker);
}

// Test 10: Stress test
void test_stress() {
    print_header("Test 10: Stress Test");
    
    RequestQueue req_queue;
    ResponseQueue resp_queue;
    std::atomic<int> total_processed(0);
    const int num_workers = 16;
    const int requests_per_worker = 100;
    
    auto start_time = std::chrono::steady_clock::now();
    
    // GPU thread
    std::thread gpu_thread([&]() {
        int processed = 0;
        while (processed < num_workers * requests_per_worker) {
            auto batch = req_queue.try_pop_batch(64, 5);
            for (const auto& req : batch) {
                ActionResponse resp(req.request_id, 0, 1.0f);
                resp_queue.push(req.request_id, resp);
                processed++;
            }
        }
    });
    
    // Worker threads
    std::vector<std::thread> workers;
    for (int i = 0; i < num_workers; i++) {
        workers.emplace_back([&, i]() {
            for (int j = 0; j < requests_per_worker; j++) {
                int req_id = i * 1000000 + j;
                StateRequest req(req_id, i, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
                req_queue.push(req);
                
                ActionResponse resp;
                if (resp_queue.try_pop(req_id, resp, 50)) {
                    total_processed++;
                }
            }
        });
    }
    
    for (auto& w : workers) {
        w.join();
    }
    gpu_thread.join();
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    print_test("All requests processed", 
               total_processed == num_workers * requests_per_worker);
    
    std::cout << YELLOW << "  Processed " << total_processed 
              << " requests in " << duration.count() << "ms" 
              << " (" << (total_processed * 1000.0 / duration.count()) 
              << " req/sec)" << RESET << std::endl;
}

int main() {
    std::cout << "\n" << BLUE << "╔════════════════════════════════════════╗" << RESET << std::endl;
    std::cout << BLUE << "║  Queue Implementation Test Suite      ║" << RESET << std::endl;
    std::cout << BLUE << "╚════════════════════════════════════════╝" << RESET << "\n" << std::endl;
    
    try {
        test_request_queue_basic();
        test_request_queue_timeout();
        test_request_queue_concurrent_push();
        test_request_queue_batch();
        test_request_queue_capacity();
        test_response_queue_basic();
        test_response_queue_timeout();
        test_response_queue_concurrent();
        test_end_to_end_flow();
        test_stress();
        
        std::cout << "\n" << GREEN << "All tests passed! ✓" << RESET << "\n" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "\n" << RED << "Test failed with exception: " 
                  << e.what() << RESET << std::endl;
        return 1;
    }
}