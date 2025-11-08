#ifndef REQUEST_QUEUE_H
#define REQUEST_QUEUE_H

#include "data_structures.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <chrono>


class RequestQueue {
public:
    explicit RequestQueue(size_t max_capacity = 1000);
    ~RequestQueue() = default;
    
    // Disable copy, allow move
    RequestQueue(const RequestQueue&) = delete;
    RequestQueue& operator=(const RequestQueue&) = delete;
    RequestQueue(RequestQueue&&) = default;
    RequestQueue& operator=(RequestQueue&&) = default;
    
    /**
     * Push a state request onto the queue (thread-safe).
     * Blocks if queue is at max capacity.
     * 
     * @param req The StateRequest to add
     */
    void push(const StateRequest& req);
    
    /**
     * Try to pop a single request with timeout.
     * 
     * @param req Output parameter for the popped request
     * @param timeout_ms Maximum time to wait in milliseconds
     * @return true if request was popped, false if timeout occurred
     */
    bool try_pop(StateRequest& req, int timeout_ms);
    
    /**
     * Pop a batch of requests (optimized for GPU processing).
     * Waits up to timeout_ms for at least one request.
     * 
     * @param max_size Maximum number of requests to pop
     * @param timeout_ms Maximum time to wait in milliseconds
     * @return Vector of requests (may be empty if timeout)
     */
    std::vector<StateRequest> try_pop_batch(size_t max_size, int timeout_ms);
    
    /**
     * Get current queue size (thread-safe).
     */
    size_t size() const;
    
    /**
     * Check if queue is empty (thread-safe).
     */
    bool empty() const;
    
    /**
     * Clear all requests from queue (thread-safe).
     */
    void clear();

private:
    std::queue<StateRequest> queue_;
    mutable std::mutex mutex_;           // Protects queue_
    std::condition_variable cv_not_empty_;  // Signals when queue has data
    std::condition_variable cv_not_full_;   // Signals when queue has space
    size_t max_capacity_;
};

#endif // REQUEST_QUEUE_H