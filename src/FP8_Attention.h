





#ifndef ATTENTION_KERNELS_H
#define ATTENTION_KERNELS_H


#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cfloat> 
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include "quantize.h"
#include <random> 
#include "../../cuda-code-repo-Sahil-Athalye/libgputk/gputk.h"


// Define block size parameters - tuned for specific hardware
#define BLOCK_SIZE_M 64  // Tile size for batch and head dimensions
#define BLOCK_SIZE_N 64  // Tile size for sequence length
#define BLOCK_SIZE_K 32  // Tile size for embedding dimension
#define WARP_SIZE 32


// Choice of attention implementation for reference GPU test
enum attention_strategy { tiled, untiled };


/**
 * FP8 FlashAttention-style tiled kernel
 */
__global__ void fp8_flash_attention_kernel(
    const half* __restrict__ query,  // [B, H, L, D]
    const half* __restrict__ key,    // [B, H, L, D]
    const half* __restrict__ value,  // [B, H, L, D]
    half*       __restrict__ output, // [B, H, L, D]
    const QuantParams quant_params,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale  // 1.0f / sqrtf(head_dim)
);

/**
 * FP32 FlashAttention-style tiled kernel
 */
__global__ void fp32_attention_kernel(
    const float* __restrict__ query,  // [B, H, L, D]
    const float* __restrict__ key,    // [B, H, L, D]
    const float* __restrict__ value,  // [B, H, L, D]
    float*       __restrict__ output, // [B, H, L, D]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale  // 1.0f / sqrtf(head_dim)
);

/**
 * Naive FP32 attention kernel (one thread per output element)
 */
__global__ void fp32_naive_attention_kernel(
    const float* __restrict__ query,  // [B, H, L, D]
    const float* __restrict__ key,    // [B, H, L, D]
    const float* __restrict__ value,  // [B, H, L, D]
    float*       __restrict__ output, // [B, H, L, D]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
);

/**
 * Compute quantization parameters for FP8 attention
 */
void compute_quant_params(
    const half* query,
    const half* key,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    QuantParams& params
);

/**
 * Host wrapper to launch the FP8 attention kernel
 */
void fp8_quantized_attention(
    const half* query,
    const half* key,
    const half* value,
    half*       output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
);

/**
 * Reference CPU implementation (FP32) for correctness checking
 */
void reference_attention(
    const float* query,
    const float* key,
    const float* value,
    float*       output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
);

/**
 * Reference GPU test launcher, selects tiled or untiled by strategy
 */
void reference_gpu_attention(
    const float*          query,
    const float*          key,
    const float*          value,
    float*                output,
    int                   batch_size,
    int                   num_heads,
    int                   seq_len,
    int                   head_dim,
    attention_strategy    strategy
);


#endif // ATTENTION_KERNELS_H
