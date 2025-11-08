#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <array>
#include <cstdint>

//  5 nodes, each 0 or 1 for compromised status
using State = std::array<float, 5>;

// Request from worker thread to RL agent
struct StateRequest {
    int request_id;      // Unique identifier for this request
    int worker_id;       // Which worker thread sent this
    State state;         // Current environment state

    StateRequest() : request_id(0), worker_id(0), state{} {}

    StateRequest(int req_id, int w_id, const State& s)
        : request_id(req_id), worker_id(w_id), state(s) {}
};

// Response from RL agent to worker thread
struct ActionResponse {
    int request_id;      // Match to original request
    int action;          // Action to take 
    float confidence;    // Confidence score 

    ActionResponse() : request_id(0), action(0), confidence(0.0f) {}

    ActionResponse(int req_id, int act, float conf)
        : request_id(req_id), action(act), confidence(conf) {}
};

#endif // DATA_STRUCTURES_H
