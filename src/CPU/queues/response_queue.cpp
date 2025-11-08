#include "response_queue.h"

void ResponseQueue::push(int request_id, const ActionResponse& resp) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Store the response in the map
    responses_[request_id] = resp;
    
    // Notify all waiting threads (they'll check if their response arrived)
    cv_.notify_all();
}

bool ResponseQueue::try_pop(int request_id, ActionResponse& resp, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Wait until our specific response arrives or timeout
    bool found = cv_.wait_for(
        lock,
        std::chrono::milliseconds(timeout_ms),
        [this, request_id]() { 
            return responses_.find(request_id) != responses_.end(); 
        }
    );
    
    if (!found) {
        return false;  // Timeout - our response never arrived
    }
    
    // Get the response and remove it from map
    auto it = responses_.find(request_id);
    resp = it->second;
    responses_.erase(it);
    
    return true;
}

bool ResponseQueue::has_response(int request_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return responses_.find(request_id) != responses_.end();
}

size_t ResponseQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return responses_.size();
}

void ResponseQueue::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    responses_.clear();
    cv_.notify_all();  // Wake up any waiting threads
}