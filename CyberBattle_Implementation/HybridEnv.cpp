/************************************************************
 * Implementation of hybrid CPU/GPU environment.
 * Combines strategic CUDA simulation with tactical CyberBattle.
 ************************************************************/

#include "HybridEnv.hpp"
#include <iostream>
#include <algorithm>
#include <stdexcept>

namespace cyber_rl {

// Constructor & Destructor
HybridEnv::HybridEnv(const HybridConfig& config)
    : config_(config)
    , current_agent_position_(0)
    , episode_step_(0)
    , last_strategic_reward_(0.0f)
    , last_tactical_reward_(0.0f)
{
    std::cout << "[HybridEnv] Initializing hybrid environment...\n";
    
    // Create strategic layer (CUDA)
    strategic_env_ = std::make_unique<CudaNetworkEnv>(config_.cuda_config);
    
    // Create tactical layer pool (CyberBattle instances)
    int num_tactical_instances = config_.num_tactical_nodes / 10;  // 10 nodes per instance
    for (int i = 0; i < num_tactical_instances; i++) {
        tactical_envs_.push_back(
            std::make_unique<CyberBattleEnv>(config_.cyberbattle_config)
        );
    }
    
    // Initialize node mapping
    initialize_node_mapping();
    
    // Activate tactical layer on critical nodes
    activate_tactical_layer();
    
    std::cout << "[HybridEnv] Initialized with:\n"
              << "  - Strategic: " << config_.cuda_config.num_nodes << " nodes (GPU)\n"
              << "  - Tactical: " << config_.num_tactical_nodes << " critical nodes (Python)\n"
              << "  - Instances: " << tactical_envs_.size() << " CyberBattle environments\n";
}

HybridEnv::~HybridEnv() {
    // Unique pointers handle cleanup automatically
}

// Initialization
void HybridEnv::initialize_node_mapping() {
    int total_nodes = config_.cuda_config.num_nodes;
    node_mapping_.resize(total_nodes);
    
    // Mark core nodes as critical
    for (int i = 0; i < total_nodes; i++) {
        node_mapping_[i].cuda_node_id = i;
        node_mapping_[i].cyberbattle_instance = -1;  // No tactical by default
        node_mapping_[i].is_critical = (i < config_.cuda_config.core_nodes);
    }
    
    std::cout << "[HybridEnv] Mapped " << config_.cuda_config.core_nodes 
              << " critical nodes\n";
}

void HybridEnv::activate_tactical_layer() {
    int activated = 0;
    int instance_idx = 0;
    
    // Activate tactical simulation for critical nodes
    for (int i = 0; i < config_.cuda_config.core_nodes && activated < config_.num_tactical_nodes; i++) {
        if (node_mapping_[i].is_critical) {
            node_mapping_[i].cyberbattle_instance = instance_idx;
            activated++;
            
            // Cycle through available instances
            instance_idx = (instance_idx + 1) % tactical_envs_.size();
        }
    }
    
    std::cout << "[HybridEnv] Activated tactical layer for " 
              << activated << " nodes\n";
}


// IEnv Interface Implementation
State HybridEnv::reset() {
    // Reset both layers
    strategic_env_->reset();
    
    for (auto& tactical_env : tactical_envs_) {
        tactical_env->reset();
    }
    
    // Reset episode state
    episode_step_ = 0;
    last_strategic_reward_ = 0.0f;
    last_tactical_reward_ = 0.0f;
    
    // Agent starts at first critical node
    current_agent_position_ = 0;
    
    return build_hybrid_observation();
}

StepResult HybridEnv::step(int action) {
    episode_step_++;
    
    // Determine if this action should use tactical layer
    bool use_tactical = should_use_tactical(current_agent_position_);
    
    // Execute on strategic layer (always)
    StepResult strategic_result = strategic_env_->step(action);
    last_strategic_reward_ = strategic_result.reward;
    
    // Execute on tactical layer (if applicable)
    float tactical_reward = 0.0f;
    if (use_tactical) {
        int instance_id = node_mapping_[current_agent_position_].cyberbattle_instance;
        if (instance_id >= 0 && instance_id < static_cast<int>(tactical_envs_.size())) {
            int tactical_action = map_to_tactical_action(action);
            StepResult tactical_result = tactical_envs_[instance_id]->step(tactical_action);
            tactical_reward = tactical_result.reward;
        }
    }
    last_tactical_reward_ = tactical_reward;
    
    // Synchronize state between layers
    sync_tactical_to_strategic();
    
    // Blend rewards
    float hybrid_reward = compute_hybrid_reward(
        last_strategic_reward_,
        last_tactical_reward_
    );
    
    // Update agent position
    current_agent_position_ = strategic_result.next_state.position;
    
    // Build hybrid observation
    State next_state = build_hybrid_observation();
    
    // Check termination
    bool done = strategic_result.done || 
                (episode_step_ >= config_.max_episode_steps);
    
    StepResult result;
    result.next_state = next_state;
    result.reward = hybrid_reward;
    result.done = done;
    
    return result;
}

int HybridEnv::get_observation_dim() const {
    // Strategic features + tactical flag
    return strategic_env_->get_observation_dim() + 1;
}

int HybridEnv::get_action_dim() const {
    return strategic_env_->get_action_dim();
}

std::string HybridEnv::get_name() const {
    return "HybridEnv-" + std::to_string(config_.cuda_config.num_nodes);
}

// State Synchronization
void HybridEnv::sync_strategic_to_tactical() {
    // In a full implementation, would sync infection status
    // from CUDA network to CyberBattle instances
    // For now, this is handled implicitly through reset()
}

void HybridEnv::sync_tactical_to_strategic() {
    // In a full implementation, would sync compromised credentials
    // from CyberBattle back to CUDA network
    // For now, tactical results influence reward only
}

// Action Routing
bool HybridEnv::should_use_tactical(int agent_pos) const {
    if (agent_pos < 0 || agent_pos >= static_cast<int>(node_mapping_.size())) {
        return false;
    }
    
    return node_mapping_[agent_pos].cyberbattle_instance >= 0;
}

int HybridEnv::map_to_tactical_action(int strategic_action) const {
    // Map strategic actions to CyberBattle action space
    // Strategic: 0-7 move, 8 scan, 9 patch, 10 isolate
    // Tactical: More detailed action space
    
    if (strategic_action < 8) {
        // Move actions - map to tactical lateral movement
        return 7;  // Lateral move action in CyberBattle
    }
    else if (strategic_action == 8) {
        // SCAN -> Service scan
        return 1;  // Scan services
    }
    else if (strategic_action == 9) {
        // PATCH -> Patch vulnerability
        return 10; // Patch service
    }
    else if (strategic_action == 10) {
        // ISOLATE -> Enable firewall
        return 12; // Enable firewall
    }
    
    return 0;  // Default: scan network
}

// Reward Blending
float HybridEnv::compute_hybrid_reward(
    float strategic_reward,
    float tactical_reward) const
{
    // Weighted combination: 70% strategic, 30% tactical
    float strategic_weight = 1.0f - config_.tactical_reward_weight;
    float tactical_weight = config_.tactical_reward_weight;
    
    return strategic_weight * strategic_reward + 
           tactical_weight * tactical_reward;
}


// Observation Construction
State HybridEnv::build_hybrid_observation() const {
    // Get strategic observation
    State strategic_obs = strategic_env_->get_agent_observation(0);
    
    // Add tactical flag (1.0 if on critical node, 0.0 otherwise)
    float tactical_flag = should_use_tactical(current_agent_position_) ? 1.0f : 0.0f;
    
    // Combine features
    std::vector<float> hybrid_features = strategic_obs.features;
    hybrid_features.push_back(tactical_flag);
    
    return State(hybrid_features, current_agent_position_);
}


// Hybrid-Specific Queries
float HybridEnv::get_strategic_infection_rate() const {
    return strategic_env_->get_infection_ratio();
}

bool HybridEnv::is_tactical_compromised() const {
    // Check if any tactical instance has been compromised
    for (const auto& tactical_env : tactical_envs_) {
        if (tactical_env->is_attacker_win()) {
            return true;
        }
    }
    return false;
}

bool HybridEnv::is_success() const {
    // Success: Strategic layer contained AND tactical layer defended
    return strategic_env_->is_success() && !is_tactical_compromised();
}

} // namespace cyber_rl