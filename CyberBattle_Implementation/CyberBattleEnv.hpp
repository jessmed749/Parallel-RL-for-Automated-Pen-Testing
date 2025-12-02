/************************************************************
 * Wrapper for CyberBattleSim Python environment.
 * Provides tactical-level cyber attack/defense simulation
 * with credential theft, service exploits, and lateral movement.
 ************************************************************/

#ifndef CYBERBATTLE_ENV_HPP
#define CYBERBATTLE_ENV_HPP

#include "IEnv.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <memory>

namespace py = pybind11;

namespace cyber_rl {
// CONFIGURATION
struct CyberBattleConfig {
    int num_nodes = 100;        // Small tactical network
    int num_subnets = 2;        // Network segmentation
    int seed = 42;              // Reproducibility
    int max_episode_steps = 500;
};


// CYBERBATTLE ENVIRONMENT
/**
 * CyberBattleEnv - Tactical cyber defense environment.
 * 
 * Wraps eisting CyberBattle Python bridge to provide
 * detailed simulation of:
 * - Credential-based attacks
 * - Service-specific exploits (SMB, RDP, SSH)
 * - Lateral movement
 * - Privilege escalation
 */
class CyberBattleEnv : public IEnv {
public:
    explicit CyberBattleEnv(const CyberBattleConfig& config = CyberBattleConfig());
    ~CyberBattleEnv() override;
    
    // Disable copy (Python objects can't be trivially copied)
    CyberBattleEnv(const CyberBattleEnv&) = delete;
    CyberBattleEnv& operator=(const CyberBattleEnv&) = delete;
    
// IEnv Interface Implementation
    State reset() override;
    StepResult step(int action) override;
    int get_observation_dim() const override;
    int get_action_dim() const override;
    std::string get_name() const override;
    
    
 // Environment Queries
    
    // Check if attacker has won (compromised all nodes)
    bool is_attacker_win() const;
    
    // Check if defender has won (stopped attack)
    bool is_defender_win() const;
    
    // Get current episode step count
    int get_episode_steps() const { return episode_step_; }
    
    
private:
    
// Python Bridge Management
    void initialize_python();
    void cleanup_python();
    
    

// State Conversion
    
    // Convert Python numpy array to C++ vector
    std::vector<float> numpy_to_vector(py::array_t<float> arr);
    
    // Convert Python observation to State struct
    State convert_observation(py::object obs);
    
    
// Member Variables    
    CyberBattleConfig config_;
    
    // Python environment instance
    py::object env_;
    py::module_ wrapper_module_;
    
    // Episode tracking
    int episode_step_;
    bool done_;
    
    // Dimensions (cached from Python)
    int observation_dim_;
    int action_dim_;
    
    // Python interpreter management
    static int python_ref_count_;
    static bool python_initialized_;
};

} // namespace cyber_rl

#endif // CYBERBATTLE_ENV_HPP