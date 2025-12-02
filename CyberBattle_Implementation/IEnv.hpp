/************************************************************
 * FILE 1: IEnv.hpp
 *
 * Minimal RL environment interface.
 * Only essential features - no optional methods.
 ************************************************************/

#ifndef IENV_HPP
#define IENV_HPP

#include <vector>
#include <string>

namespace cyber_rl
{

    /**
     * State representation - minimal observation for agent.
     */
    struct State
    {
        std::vector<float> features; // Flattened observation vector
        int position;                // Agent's current node (for graph envs)

        State() : position(0) {}
        State(const std::vector<float> &f, int pos) : features(f), position(pos) {}
    };

    /**
     * Result of taking a step in the environment.
     */
    struct StepResult
    {
        State next_state; // Observation after action
        float reward;     // Immediate reward
        bool done;        // Episode finished (terminal or timeout)

        StepResult() : reward(0.0f), done(false) {}
    };

    // ABSTRACT ENVIRONMENT INTERFACE

    /**
     * IEnv - Minimal interface for RL environments.
     *
     * Implementations:
     * - CudaNetworkEnv: GPU-accelerated 50k node network
     * - CyberBattleEnv: CyberBattleSim tactical wrapper
     * - HybridEnv: Combined CUDA + CyberBattle
     */
    class IEnv
    {
    public:
        virtual ~IEnv() = default;

        // Reset environment to initial state
        virtual State reset() = 0;

        // Take action, return (next_state, reward, done)
        virtual StepResult step(int action) = 0;

        // Get observation space size
        virtual int get_observation_dim() const = 0;

        // Get action space size
        virtual int get_action_dim() const = 0;

        // Get environment name for logging
        virtual std::string get_name() const = 0;
    };

} // namespace cyber_rl

#endif // IENV_HPP