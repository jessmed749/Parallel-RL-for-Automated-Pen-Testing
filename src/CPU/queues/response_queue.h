#ifndef RESPONSE_QUEUE_H
#define RESPONSE_QUEUE_H

#include "data_structures.h"
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <chrono>

class ResponseQueue {
public:
    ResponseQueue() = default;
    ~ResponseQueue() = default;
    
    // Disable copy, allow move
    ResponseQueue(const ResponseQueue&) = delete;
    ResponseQueue& operator=(const ResponseQueue&) = delete;
    ResponseQueue(ResponseQueue&&) = default;
    ResponseQueue& operator=(ResponseQueue&&) = default;
    
    /**
     * Push an action response (called by GPU thread).
     * 
     * @param request_id The request ID to store response under
     * @param resp The ActionResponse to store
     */
    void push(int request_id, const ActionResponse& resp);
    
    /**
     * Try to pop a specific response by request_id.
     * Blocks until the response is available or timeout occurs.
     * 
     * @param request_id The ID of the request to get response for
     * @param resp Output parameter for the response
     * @param timeout_ms Maximum time to wait in milliseconds
     * @return true if response was found, false if timeout
     */
    bool try_pop(int request_id, ActionResponse& resp, int timeout_ms);
    
    /**
     * Check if a response for a specific request_id exists.
     * 
     * @param request_id The ID to check
     * @return true if response exists
     */
    bool has_response(int request_id) const;
    
    /**
     * Get current number of pending responses.
     */
    size_t size() const;
    
    /**
     * Clear all responses from queue (thread-safe).
     */
    void clear();

private:
    std::unordered_map<int, ActionResponse> responses_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;  // Notifies when new responses arrive
};

#endif // RESPONSE_QUEUE_H