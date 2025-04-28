

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