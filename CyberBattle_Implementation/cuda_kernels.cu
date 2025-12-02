/************************************************************
 * GPU kernels for network vulnerability propagation.
 * Implements SIR (Susceptible-Infected-Recovered) model
 * with isolation support on 50k node network.
 ************************************************************/

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>


// KERNEL 1: Initialize RNG States
__global__ void init_rng_kernel(
    curandState* states,
    int num_nodes,
    unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}


// KERNEL 2: SIR Propagation with Isolation
__global__ void sir_propagation_kernel(
    const int* __restrict__ d_offsets,
    const int* __restrict__ d_neighbors,
    const int* __restrict__ d_state,
    const int* __restrict__ d_isolated,
    int* d_next_state,
    int num_nodes,
    int max_neighbors,
    float p_infect,
    float p_recover,
    curandState* rng_states)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    
    int current_state = d_state[v];
    int is_isolated = d_isolated[v];
    
    // Default: keep current state
    int next_state = current_state;
    
    // RECOVERED stays RECOVERED (permanent immunity)
    if (current_state == 2) {
        d_next_state[v] = 2;
        return;
    }
    
    // Get thread-local RNG
    curandState local_rng = rng_states[v];
    
    // INFECTED -> attempt recovery
    if (current_state == 1) {
        float r = curand_uniform(&local_rng);
        if (r < p_recover) {
            next_state = 2;  // Recovered
        } else {
            next_state = 1;  // Remain infected
        }
    }
    
    // SUSCEPTIBLE -> check for infection from neighbors
    if (current_state == 0 && !is_isolated) {
        int start = d_offsets[v];
        int end = start + max_neighbors;
        
        // Check each neighbor
        for (int i = start; i < end; i++) {
            int neighbor = d_neighbors[i];
            
            // Skip if neighbor is isolated or not infected
            if (d_isolated[neighbor] || d_state[neighbor] != 1) {
                continue;
            }
            
            // Attempt infection
            float r = curand_uniform(&local_rng);
            if (r < p_infect) {
                next_state = 1;  // Infected!
                break;
            }
        }
    }
    
    // Write back RNG state
    rng_states[v] = local_rng;
    
    // Write next state
    d_next_state[v] = next_state;
}


// KERNEL 3: Copy State Buffer
__global__ void copy_state_kernel(
    int* d_dest,
    const int* d_src,
    int num_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        d_dest[idx] = d_src[idx];
    }
}

// KERNEL 4: Q-Learning Update
__global__ void q_learning_update_kernel(
    const int* __restrict__ d_states,
    const int* __restrict__ d_actions,
    const float* __restrict__ d_rewards,
    const int* __restrict__ d_next_states,
    const bool* __restrict__ d_dones,
    float* d_q_table,
    int batch_size,
    int num_actions,
    float alpha,
    float gamma)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    int state = d_states[idx];
    int action = d_actions[idx];
    float reward = d_rewards[idx];
    int next_state = d_next_states[idx];
    bool done = d_dones[idx];
    
    // Find max Q(s', a')
    float max_next_q = 0.0f;
    if (!done) {
        int next_base = next_state * num_actions;
        max_next_q = d_q_table[next_base];
        
        #pragma unroll
        for (int a = 1; a < num_actions; a++) {
            float q = d_q_table[next_base + a];
            if (q > max_next_q) {
                max_next_q = q;
            }
        }
    }
    
    // Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
    int q_idx = state * num_actions + action;
    float old_q = d_q_table[q_idx];
    float target = reward + gamma * max_next_q;
    float new_q = old_q + alpha * (target - old_q);
    
    d_q_table[q_idx] = new_q;
}



// HOST WRAPPER FUNCTIONS (Callable from CudaNetworkEnv.cpp)
extern "C" {

/**
 * Initialize RNG states for all nodes.
 */
void launch_init_rng(
    curandState* d_rng_states,
    int num_nodes,
    unsigned long seed)
{
    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;
    
    init_rng_kernel<<<blocks, threads>>>(d_rng_states, num_nodes, seed);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in init_rng: %s\n", cudaGetErrorString(err));
    }
}


/**
 * Run one step of SIR propagation across the network.
 */
void launch_sir_propagation(
    const int* d_offsets,
    const int* d_neighbors,
    const int* d_state,
    const int* d_isolated,
    int* d_next_state,
    int num_nodes,
    int max_neighbors,
    float p_infect,
    float p_recover,
    curandState* d_rng_states)
{
    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;
    
    sir_propagation_kernel<<<blocks, threads>>>(
        d_offsets,
        d_neighbors,
        d_state,
        d_isolated,
        d_next_state,
        num_nodes,
        max_neighbors,
        p_infect,
        p_recover,
        d_rng_states
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in sir_propagation: %s\n", cudaGetErrorString(err));
    }
}


/**
 * Copy next_state buffer to state buffer.
 */
void launch_copy_state(
    int* d_state,
    const int* d_next_state,
    int num_nodes)
{
    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;
    
    copy_state_kernel<<<blocks, threads>>>(d_state, d_next_state, num_nodes);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in copy_state: %s\n", cudaGetErrorString(err));
    }
}


/**
 * Update Q-table using batch of transitions.
 */
void launch_q_learning_update(
    const int* d_states,
    const int* d_actions,
    const float* d_rewards,
    const int* d_next_states,
    const bool* d_dones,
    float* d_q_table,
    int batch_size,
    int num_actions,
    float alpha,
    float gamma)
{
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    q_learning_update_kernel<<<blocks, threads>>>(
        d_states,
        d_actions,
        d_rewards,
        d_next_states,
        d_dones,
        d_q_table,
        batch_size,
        num_actions,
        alpha,
        gamma
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in q_learning_update: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"