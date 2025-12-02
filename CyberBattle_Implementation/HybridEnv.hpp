/************************************************************
 * Hybrid CPU/GPU environment combining:
 * - Strategic layer: CudaNetworkEnv (50k nodes, GPU propagation)
 * - Tactical layer: CyberBattleEnv (100 critical nodes, detailed attacks)
 * 
 * Demonstrating how CPU coordinates high-level strategy while GPU handles
 * massive-scale propagation, with tactical detail where needed.
 ************************************************************/

#ifndef HYBRID_ENV_HPP
#define HYBRID_ENV_HPP

#include "IEnv.hpp"
#include "CudaNetworkEnv.hpp"
#include "CyberBattleEnv.hpp"
#include <memory>
#include <vector>

namespace cyber_rl {
// CONFIGURATION
struct HybridConfig {
    // Strategic layer (CUDA)
    CudaEnvConfig cuda_config;
    
    // Tactical layer (CyberBattle)
    CyberBattleConfig cyberbattle_config;
    
    // Integration settings
    int num_tactical_nodes = 100;      // How many nodes get detailed simulation
    float tactical_reward_weight = 0.3f; // Blend: 70% strategic, 30% tactical
    
    // Episode control
    int max_episode_steps = 500;
};



// NODE MAPPING
struct NodeMapping {
    int cuda_node_id;           // Node ID in strategic network
    int cyberbattle_instance;   // Which CyberBattle instance (-1 if none)
    bool is_critical;           // Is this a critical node?
};


/**
 * HybridEnv - Two-layer defense simulation.
 * 
 * Architecture:
 * - Strategic layer: 50k nodes, GPU SIR propagation, fast
 * - Tactical layer: 100 critical nodes, detailed Python sim, realistic
 * - CPU coordination: Routes actions, blends rewards, syncs state
 */
class HybridEnv : public IEnv {
public:
    explicit HybridEnv(const HybridConfig& config = HybridConfig());
    ~HybridEnv() override;
    
    // Disable copy
    HybridEnv(const HybridEnv&) = delete;
    HybridEnv& operator=(const HybridEnv&) = delete;
    
    

// IEnv Interface Implementation  
    State reset() override;
    StepResult step(int action) override;
    int get_observation_dim() const override;
    int get_action_dim() const override;
    std::string get_name() const override;
    
    
    
// Get strategic layer infection rate
    float get_strategic_infection_rate() const;
    
// Get tactical layer status
    bool is_tactical_compromised() const;
    
// Check if episode succeeded (contained outbreak)
    bool is_success() const;
    
    
private:

// Initialization
    void initialize_node_mapping();
    void activate_tactical_layer();
    
    
 
// Sync strategic state → tactical instances
    void sync_strategic_to_tactical();
    
// Sync tactical state → strategic layer
    void sync_tactical_to_strategic();
    
    

// Determine if action should go to tactical layer
    bool should_use_tactical(int agent_pos) const;
    
// Map strategic action to tactical action
    int map_to_tactical_action(int strategic_action) const;
  
    

// Combine strategic and tactical rewards
    float compute_hybrid_reward(
        float strategic_reward,
        float tactical_reward
    ) const;
    

// Build hybrid observation (strategic + tactical features)
    State build_hybrid_observation() const;
    
    

// Member Variables  
    HybridConfig config_;
    
    // Two-layer architecture
    std::unique_ptr<CudaNetworkEnv> strategic_env_;
    std::vector<std::unique_ptr<CyberBattleEnv>> tactical_envs_;
    
    // Node mapping: which CUDA nodes have tactical detail
    std::vector<NodeMapping> node_mapping_;
    
    // Current agent state
    int current_agent_position_;
    
    // Episode tracking
    int episode_step_;
    
    // Last rewards (for blending)
    float last_strategic_reward_;
    float last_tactical_reward_;
};

} // namespace cyber_rl

#endif // HYBRID_ENV_HPP