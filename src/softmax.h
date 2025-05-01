

#ifndef SOFTMAX_H
#define SOFTMAX_H
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Online softmax implementation for processing a block at a time
// This avoids materializing the full attention matrix
__device__ void online_softmax(half* block_result, float& m_prev, float& m_curr, 
                             float& d_prev, float& d_curr, int block_size) {
    // Find maximum value in current block (m_curr)
    float max_val = -INFINITY;
    for (int i = 0; i < block_size; i++) {
        max_val = fmaxf(max_val, __half2float(block_result[i]));
    }
    m_curr = max_val;
    
    // Compute scale factors for numerical stability
    float m_diff = m_curr - m_prev;
    float scale = expf(m_prev - m_curr);
    
    // Update denominator and scale previous blocks
    if (m_diff != 0.0f) {
        d_curr = d_prev * scale;
    } else {
        d_curr = d_prev;
    }
    
    // Compute softmax for current block
    for (int i = 0; i < block_size; i++) {
        float val = expf(__half2float(block_result[i]) - m_curr);
        d_curr += val;
        block_result[i] = __float2half(val);
    }
    
    // Final scaling for stable softmax
    for (int i = 0; i < block_size; i++) {
        block_result[i] = __float2half(__half2float(block_result[i]) / d_curr);
    }
}

__device__ void improved_online_softmax(
    half* block_result, 
    float& m_prev, 
    float& m_curr, 
    float& d_prev, 
    float& d_curr, 
    int block_size,
    bool is_first_block = false) {
    
    // Initialize values for first block
    if (is_first_block) {
        m_prev = -INFINITY;
        d_prev = 0.0f;
    }
    
    // Find maximum value in current block
    float local_max = -INFINITY;
    for (int i = 0; i < block_size; i++) {
        local_max = fmaxf(local_max, __half2float(block_result[i]));
    }
    
    // Determine global maximum considering previous blocks
    float global_max = fmaxf(m_prev, local_max);
    m_curr = global_max;
    
    // Skip unnecessary computations if block is empty
    if (block_size == 0) {
        d_curr = d_prev;
        return;
    }
    
    // Scale factor for previous denominator
    float scale_prev = (m_prev != -INFINITY) ? expf(m_prev - global_max) : 0.0f;
    
    // Update denominator with scaled previous value
    d_curr = d_prev * scale_prev;
    
    // Compute softmax for current block
    for (int i = 0; i < block_size; i++) {
        float val = expf(__half2float(block_result[i]) - global_max);
        d_curr += val;
        block_result[i] = __float2half(val); // Store unnormalized values temporarily
    }
    
    // Perform final normalization
    if (d_curr > 1e-20f) { // Avoid division by very small values
        for (int i = 0; i < block_size; i++) {
            block_result[i] = __float2half(__half2float(block_result[i]) / d_curr);
        }
    } else {
        // Handle degenerate case with uniform distribution
        float uniform_val = 1.0f / block_size;
        for (int i = 0; i < block_size; i++) {
            block_result[i] = __float2half(uniform_val);
        }
    }
}

// Online softmax implementation for processing a block at a time
// This avoids materializing the full attention matrix
__device__ void improved_online_softmax_two(half* block_result, float& m_prev, float& m_curr, 
                               float& d_prev, float& d_curr, int block_size) {
    // Find maximum value in current block (m_curr)
    float max_val = -INFINITY;
    for (int i = 0; i < block_size; i++) {
        max_val = fmaxf(max_val, __half2float(block_result[i]));
    }
    
    // Use thread-block reduction if in a CUDA context (simplified here)
    // In the original, this would use quad_allreduce_ or similar
    m_curr = max_val;
    
    // Scale factor using log2 domain for better FFMA utilization
    // Using softmax_scale similar to the original (typically 1.0/sqrt(head_dim))
    const float softmax_scale_log2 = 1.44269504f; // M_LOG2E
    
    // Handle edge case when max is -infinity (all inputs were -infinity)
    const float m_scaled = m_curr == -INFINITY ? 0.0f : m_curr * softmax_scale_log2;
    
    // Only rescale previous results if this isn't the first block
    if (d_prev > 0.0f) {
        // Using exp2f instead of expf for better performance
        float scale = exp2f((m_prev - m_curr) * softmax_scale_log2);
        d_curr = d_prev * scale;
    } else {
        d_curr = 0.0f;
    }
    
    // Compute exponentials for current block
    for (int i = 0; i < block_size; i++) {
        // Using exp2f and FFMA-friendly computation
        float val = exp2f(__half2float(block_result[i]) * softmax_scale_log2 - m_scaled);
        d_curr += val;
        block_result[i] = __float2half(val);
    }
    
    // Numerical stability checks for the sum
    float inv_sum = (d_curr == 0.0f || isnan(d_curr)) ? 1.0f : 1.0f / d_curr;
    
    // Final normalization
    for (int i = 0; i < block_size; i++) {
        block_result[i] = __float2half(__half2float(block_result[i]) * inv_sum);
    }
}

// Online softmax implementation for processing a block at a time
// This avoids materializing the full attention matrix
__device__ void online_softmax_full(float * block_result, float& m_prev, float& m_curr, 
                             float& d_prev, float& d_curr, int block_size) {
    // Find maximum value in current block (m_curr)
    float max_val = -INFINITY;
    for (int i = 0; i < block_size; i++) {
        max_val = fmaxf(max_val, block_result[i]);
    }
    m_curr = max_val;
    
    // Compute scale factors for numerical stability
    float m_diff = m_curr - m_prev;
    float scale = expf(m_prev - m_curr);
    
    // Update denominator and scale previous blocks
    if (m_diff != 0.0f) {
        d_curr = d_prev * scale;
    } else {
        d_curr = d_prev;
    }
    
    // Compute softmax for current block
    for (int i = 0; i < block_size; i++) {
        float val = expf(block_result[i] - m_curr);
        d_curr += val;
        block_result[i] = val;
    }
    
    // Final scaling for stable softmax
    for (int i = 0; i < block_size; i++) {
        block_result[i] = block_result[i] / d_curr;
    }
}

#endif