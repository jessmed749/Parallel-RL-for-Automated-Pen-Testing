/************************************************************
 * FILE 11: main.cpp
 *
 * Main training program for hybrid CPU/GPU RL system.
 * Demonstrates large-scale network vulnerability propagation
 * with autonomous defense agents.
 *
 * Usage:
 *   ./cyber_rl [cuda|cyberbattle|hybrid] [episodes] [--analyze]
 *
 * Examples:
 *   ./cyber_rl hybrid              # Train on hybrid env (default)
 *   ./cyber_rl cuda 10000          # Train on CUDA env for 10k episodes
 *   ./cyber_rl hybrid 5000 -a      # Train + run vulnerability analysis
 *
 * Default: hybrid environment, 5000 episodes, no analysis
 ************************************************************/

#include "IEnv.hpp"
#include "CudaNetworkEnv.hpp"
#include "CyberBattleEnv.hpp"
#include "HybridEnv.hpp"
#include "TabularQLearning.hpp"
#include "VulnerabilityAnalyzer.hpp"

#include <iostream>
#include <string>
#include <memory>
#include <chrono>

using namespace cyber_rl;

// CONFIGURATION
struct TrainingConfig
{
    std::string env_type = "hybrid"; // cuda, cyberbattle, or hybrid
    int num_episodes = 5000;
    int eval_frequency = 300;
    bool verbose = true;
    bool run_analysis = false; // NEW: Run vulnerability analysis after training
};

// ENVIRONMENT FACTORY
std::unique_ptr<IEnv> create_environment(const std::string &env_type)
{
    std::cout << "\n=== CREATING ENVIRONMENT ===\n";
    std::cout << "Type: " << env_type << "\n\n";

    if (env_type == "cuda")
    {
        // Pure CUDA environment - 50k nodes, GPU acceleration
        CudaEnvConfig config;
        config.num_nodes = 50000;
        config.num_agents = 8;
        config.max_episode_steps = 500;
        return std::make_unique<CudaNetworkEnv>(config);
    }
    else if (env_type == "cyberbattle")
    {
        // Pure CyberBattle environment - 100 nodes, tactical detail
        CyberBattleConfig config;
        config.num_nodes = 100;
        config.num_subnets = 2;
        config.max_episode_steps = 500;
        return std::make_unique<CyberBattleEnv>(config);
    }
    else if (env_type == "hybrid")
    {
        // Hybrid environment - 50k strategic + 100 tactical
        HybridConfig config;
        config.cuda_config.num_nodes = 50000;
        config.cuda_config.num_agents = 8;
        config.num_tactical_nodes = 100;
        config.tactical_reward_weight = 0.3f;
        config.max_episode_steps = 500;
        return std::make_unique<HybridEnv>(config);
    }
    else
    {
        std::cerr << "Unknown environment type: " << env_type << "\n";
        std::cerr << "Valid options: cuda, cyberbattle, hybrid\n";
        exit(1);
    }
}

// TRAINING
void train_agent(IEnv *env, const TrainingConfig &train_config)
{
    std::cout << "\n=== TRAINING CONFIGURATION ===\n";
    std::cout << "Environment: " << env->get_name() << "\n";
    std::cout << "Observation dim: " << env->get_observation_dim() << "\n";
    std::cout << "Action dim: " << env->get_action_dim() << "\n";
    std::cout << "Episodes: " << train_config.num_episodes << "\n\n";

    // Create Q-learning agent
    QLearningConfig rl_config;
    rl_config.learning_rate = 0.01f;
    rl_config.discount_factor = 0.99f;
    rl_config.epsilon = 0.1f;
    rl_config.replay_capacity = 100000;
    rl_config.min_replay_size = 1000;
    rl_config.batch_size = 256;
    rl_config.num_workers = 4;
    rl_config.max_episodes = train_config.num_episodes;
    rl_config.eval_frequency = train_config.eval_frequency;

    TabularQLearning agent(env, rl_config);

    // Train
    auto start_time = std::chrono::steady_clock::now();

    agent.train(train_config.num_episodes);

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "\n=== TRAINING COMPLETE ===\n";
    std::cout << "Total time: " << duration.count() << " seconds\n";
    std::cout << "Final metrics:\n";
    agent.print_metrics();
}

// EVALUATION
void evaluate_agent(IEnv *env, TabularQLearning *agent = nullptr)
{
    std::cout << "\n=== POLICY EVALUATION ===\n";

    // Test on multiple starting positions
    std::vector<int> test_positions = {0, 1000, 25000};

    for (int start_pos : test_positions)
    {
        State state = env->reset();
        state.position = start_pos;

        float total_reward = 0.0f;
        int steps = 0;
        bool success = false;

        for (int step = 0; step < 500; step++)
        {
            int action;
            if (agent)
            {
                // Use trained agent
                action = agent->get_best_action(state.position);
            }
            else
            {
                // Random action
                action = rand() % env->get_action_dim();
            }

            StepResult result = env->step(action);
            total_reward += result.reward;
            steps++;

            if (result.done)
            {
                success = (result.reward > 50.0f);
                break;
            }

            state = result.next_state;
        }

        std::cout << "Start position " << start_pos << ": "
                  << (success ? "SUCCESS" : "FAILED")
                  << " in " << steps << " steps, "
                  << "reward = " << total_reward << "\n";
    }
}

// VULNERABILITY ANALYSIS
void run_vulnerability_analysis(TabularQLearning *trained_agent)
{
    std::cout << "\n========================================\n";
    std::cout << "  VULNERABILITY ANALYSIS\n";
    std::cout << "========================================\n\n";

    VulnerabilityAnalyzer analyzer;

    // Part 1: Node type criticality
    std::cout << "PART 1: Analyzing node type criticality...\n\n";
    auto criticality_results = analyzer.run_comparative_analysis();
    analyzer.print_results(criticality_results);
    analyzer.export_to_csv(criticality_results, "node_criticality.csv");

    // Part 2: Defense effectiveness (if agent provided)
    if (trained_agent)
    {
        std::cout << "\nPART 2: Analyzing defense effectiveness...\n\n";
        auto defense_results = analyzer.analyze_defense_effectiveness(trained_agent);
        analyzer.print_defense_comparison(defense_results);
    }

    std::cout << "\n[Analysis] Results exported to node_criticality.csv\n";
}

int main(int argc, char **argv)
{
    std::cout << "========================================\n";
    std::cout << "  HYBRID CPU/GPU RL SYSTEM\n";
    std::cout << "  Network Vulnerability Defense\n";
    std::cout << "========================================\n";

    // Parse command line arguments
    TrainingConfig config;
    if (argc > 1)
    {
        config.env_type = argv[1];
    }

    if (argc > 2)
    {
        config.num_episodes = std::stoi(argv[2]);
    }

    if (argc > 3)
    {
        std::string analysis_flag = argv[3];
        config.run_analysis = (analysis_flag == "--analyze" || analysis_flag == "-a");
    }

    try
    {
        // Create environment
        auto env = create_environment(config.env_type);

        // Train agent
        TabularQLearning *agent_ptr = nullptr;
        {
            TabularQLearning agent(env.get(), QLearningConfig());
            train_agent(env.get(), config);

            // Evaluate policy
            evaluate_agent(env.get(), &agent);

            // Run vulnerability analysis if requested
            if (config.run_analysis)
            {
                run_vulnerability_analysis(&agent);
            }

            agent_ptr = &agent;
        }

        std::cout << "\n=== SUCCESS ===\n";
        std::cout << "Training completed successfully!\n";
        std::cout << "Environment: " << env->get_name() << "\n";

        if (config.run_analysis)
        {
            std::cout << "\nVulnerability analysis complete!\n";
            std::cout << "Check node_criticality.csv for results.\n";
        }
        else
        {
            std::cout << "\nTip: Run with '--analyze' flag for vulnerability analysis:\n";
            std::cout << "  ./cyber_rl " << config.env_type << " "
                      << config.num_episodes << " --analyze\n";
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n=== ERROR ===\n";
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}

/************************************************************
 * COMPILATION INSTRUCTIONS
 *
 * Required dependencies:
 * - CUDA Toolkit (>=11.0)
 * - pybind11 (for CyberBattle wrapper)
 * - Python 3.8+
 *
 * Compile command:
 *
 * nvcc -o cyber_rl \
 *   main.cpp \
 *   CudaNetworkEnv.cpp \
 *   CyberBattleEnv.cpp \
 *   HybridEnv.cpp \
 *   TabularQLearning.cpp \
 *   cuda_kernels.cu \
 *   -I/usr/include/python3.8 \
 *   -I/path/to/pybind11/include \
 *   -lpython3.8 \
 *   -std=c++14 \
 *   -O3 \
 *   -arch=sm_70 \
 *   -lcurand
 *
 * Run:
 *
 * # Train on hybrid environment (default)
 * ./cyber_rl hybrid
 *
 * # Train on pure CUDA (fast, 50k nodes)
 * ./cyber_rl cuda
 *
 * # Train on pure CyberBattle (realistic, 100 nodes)
 * ./cyber_rl cyberbattle
 *
 * # Custom episode count
 * ./cyber_rl hybrid 10000
 *
 ************************************************************/