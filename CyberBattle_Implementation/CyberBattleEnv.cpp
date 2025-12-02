/************************************************************
 * FILE 6: CyberBattleEnv.cpp
 * 
 * Implementation of CyberBattleSim wrapper environment.
 ************************************************************/

#include "CyberBattleEnv.hpp"
#include <iostream>
#include <stdexcept>
#include <Python.h>

namespace cyber_rl {

/////////////////////////////////////////////////////////////
// Static Member Initialization
/////////////////////////////////////////////////////////////

int CyberBattleEnv::python_ref_count_ = 0;
bool CyberBattleEnv::python_initialized_ = false;


/////////////////////////////////////////////////////////////
// Constructor / Destructor
/////////////////////////////////////////////////////////////

CyberBattleEnv::CyberBattleEnv(const CyberBattleConfig& config)
    : config_(config)
    , episode_step_(0)
    , done_(false)
    , observation_dim_(0)
    , action_dim_(0)
{
    try {
        initialize_python();
        
        // Import your existing Python wrapper module
        wrapper_module_ = py::module_::import("cyberbattle_env.env_wrapper");
        
        // Create environment instance
        py::object wrapper_class = wrapper_module_.attr("CyberBattleWrapper");
        env_ = wrapper_class(config_.num_nodes, config_.num_subnets, config_.seed);
        
        // Get dimensions from Python environment
        observation_dim_ = env_.attr("get_observation_space")().cast<int>();
        action_dim_ = env_.attr("get_action_space")().cast<int>();
        
        std::cout << "[CyberBattleEnv] Initialized with " << config_.num_nodes
                  << " nodes, obs_dim=" << observation_dim_
                  << ", action_dim=" << action_dim_ << std::endl;
        
    } catch (py::error_already_set& e) {
        python_ref_count_--;
        std::cerr << "[CyberBattleEnv] Python error: " << e.what() << std::endl;
        throw std::runtime_error("Failed to initialize CyberBattleSim");
    }
}

CyberBattleEnv::~CyberBattleEnv() {
    cleanup_python();
}


/////////////////////////////////////////////////////////////
// Python Interpreter Management
/////////////////////////////////////////////////////////////

void CyberBattleEnv::initialize_python() {
    if (!python_initialized_) {
        // Try to initialize - works with both pybind11 versions
        if (!Py_IsInitialized()) {
            Py_Initialize();
        }
        python_initialized_ = true;
        
        try {
            // Add Python module path
            py::module_ sys = py::module_::import("sys");
            py::object path = sys.attr("path");
            
            const char* python_path = std::getenv("CYBERBATTLE_PYTHON_PATH");
            if (python_path) {
                path.attr("append")(python_path);
                std::cout << "[CyberBattleEnv] Using CYBERBATTLE_PYTHON_PATH: " 
                          << python_path << std::endl;
            } else {
                path.attr("append")("/app/python");
                std::cout << "[CyberBattleEnv] Using default path: /app/python" << std::endl;
            }
            
            std::cout << "[CyberBattleEnv] Python interpreter initialized" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[CyberBattleEnv] Warning: Could not set Python path: " << e.what() << std::endl;
        }
    }
    python_ref_count_++;
}

void CyberBattleEnv::cleanup_python() {
    env_.release();
    wrapper_module_.release();
    
    python_ref_count_--;
    if (python_ref_count_ == 0 && python_initialized_) {
        // Last instance - could finalize Python here
        // py::finalize_interpreter();
        // python_initialized_ = false;
        // Note: Usually don't finalize in long-running apps
    }
}


/////////////////////////////////////////////////////////////
// IEnv Interface Implementation
/////////////////////////////////////////////////////////////

State CyberBattleEnv::reset() {
    try {
        py::object result = env_.attr("reset")();
        
        // Handle Gym API: reset() returns (obs, info) or just obs
        py::array_t<float> obs_array;
        if (py::isinstance<py::tuple>(result)) {
            py::tuple result_tuple = result.cast<py::tuple>();
            obs_array = result_tuple[0].cast<py::array_t<float>>();
        } else {
            obs_array = result.cast<py::array_t<float>>();
        }
        
        done_ = false;
        episode_step_ = 0;
        
        std::vector<float> features = numpy_to_vector(obs_array);
        return State(features, 0);  // Position = 0 (not used in tactical env)
        
    } catch (py::error_already_set& e) {
        std::cerr << "[CyberBattleEnv] Reset error: " << e.what() << std::endl;
        throw std::runtime_error("Failed to reset environment");
    }
}

StepResult CyberBattleEnv::step(int action) {
    try {
        // Call Python step method
        py::tuple result = env_.attr("step")(action).cast<py::tuple>();
        
        if (result.size() != 5) {
            throw std::runtime_error("Step returned invalid tuple size");
        }
        
        // Unpack: (observation, reward, terminated, truncated, info)
        py::array_t<float> obs_array = result[0].cast<py::array_t<float>>();
        float reward = result[1].cast<float>();
        bool terminated = result[2].cast<bool>();
        bool truncated = result[3].cast<bool>();
        
        done_ = terminated || truncated;
        episode_step_++;
        
        StepResult step_result;
        step_result.next_state = State(numpy_to_vector(obs_array), 0);
        step_result.reward = reward;
        step_result.done = done_;
        
        return step_result;
        
    } catch (py::error_already_set& e) {
        std::cerr << "[CyberBattleEnv] Step error: " << e.what() << std::endl;
        throw std::runtime_error("Failed to execute step");
    }
}

int CyberBattleEnv::get_observation_dim() const {
    return observation_dim_;
}

int CyberBattleEnv::get_action_dim() const {
    return action_dim_;
}

std::string CyberBattleEnv::get_name() const {
    return "CyberBattleEnv-" + std::to_string(config_.num_nodes);
}


/////////////////////////////////////////////////////////////
// Environment Queries
/////////////////////////////////////////////////////////////

bool CyberBattleEnv::is_attacker_win() const {
    // Could query Python environment for attacker success
    // For now, delegate to done_ flag
    return done_;
}

bool CyberBattleEnv::is_defender_win() const {
    // Could query Python environment for defender success
    return false;  // Simplified for now
}


/////////////////////////////////////////////////////////////
// State Conversion Utilities
/////////////////////////////////////////////////////////////

std::vector<float> CyberBattleEnv::numpy_to_vector(py::array_t<float> arr) {
    py::buffer_info buf = arr.request();
    float* ptr = static_cast<float*>(buf.ptr);
    return std::vector<float>(ptr, ptr + buf.size);
}

State CyberBattleEnv::convert_observation(py::object obs) {
    py::array_t<float> arr = obs.cast<py::array_t<float>>();
    std::vector<float> features = numpy_to_vector(arr);
    return State(features, 0);
}

} // namespace cyber_rl


/************************************************************
 * USAGE NOTES
 * 
 * This wrapper integrates your existing cyberbattle_wrapper.cpp
 * Python bridge into the IEnv interface.
 * 
 * Key Integration Points:
 * 1. Python interpreter managed via reference counting
 * 2. Environment lifecycle: reset() → step() × N → done
 * 3. Observations converted from NumPy → std::vector<float>
 * 
 * Example Usage:
 *   CyberBattleEnv env;
 *   State s = env.reset();
 *   StepResult result = env.step(action);
 * 
 * Performance:
 * - Python overhead: ~1000 steps/sec (vs 100k for CUDA)
 * - Suitable for tactical simulation (100 nodes)
 * - Complements CUDA environment (realism vs scale)
 ************************************************************/