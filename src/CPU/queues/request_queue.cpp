#include "request_queue.h"

RequestQueue::RequestQueue(size_t max_capacity) 
    : max_capacity_(max_capacity) {
}

void RequestQueue::push(const StateRequest& req) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Wait if queue is full
    cv_not_full_.wait(lock, [this]() { 
        return queue_.size() < max_capacity_; 
    });
    
    queue_.push(req);
    
    // Notify waiting threads that queue has data
    cv_not_empty_.notify_one();
}

bool RequestQueue::try_pop(StateRequest& req, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Wait for data or timeout
    bool has_data = cv_not_empty_.wait_for(
        lock,
        std::chrono::milliseconds(timeout_ms),
        [this]() { return !queue_.empty(); }
    );
    
    if (!has_data) {
        return false;  // Timeout occurred
    }
    
    // Pop the request
    req = queue_.front();
    queue_.pop();
    
    // Notify that queue has space
    cv_not_full_.notify_one();
    
    return true;
}

std::vector<StateRequest> RequestQueue::try_pop_batch(size_t max_size, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Wait for at least one request or timeout
    bool has_data = cv_not_empty_.wait_for(
        lock,
        std::chrono::milliseconds(timeout_ms),
        [this]() { return !queue_.empty(); }
    );
    
    std::vector<StateRequest> batch;
    
    if (!has_data) {
        return batch;  // Return empty batch on timeout
    }
    
    // Pop up to max_size requests
    while (!queue_.empty() && batch.size() < max_size) {
        batch.push_back(queue_.front());
        queue_.pop();
    }
    
    // Notify that queue has space 
    if (batch.size() > 0) {
        cv_not_full_.notify_all();
    }
    
    return batch;
}

size_t RequestQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

bool RequestQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

void RequestQueue::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Clear the queue
    while (!queue_.empty()) {
        queue_.pop();
    }
    
    // Notify all threads waiting to push
    cv_not_full_.notify_all();
}