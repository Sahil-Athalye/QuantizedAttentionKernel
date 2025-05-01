#include "FP8_Attention.h"
#include "softmax.h"

/**
 * Main kernel for FP8 quantized attention with FlashAttention-style tiling
 * Dimensions:
 * - batch_size: number of sequences in the batch
 * - num_heads: number of attention heads
 * - seq_len: sequence length
 * - head_dim: dimensionality of each attention head
 */
__global__ void fp8_flash_attention_kernel(
    const half
        *__restrict__ query, // [batch_size, num_heads, seq_len, head_dim]
    const half
        *__restrict__ key, // [batch_size, num_heads, seq_len, head_dim]
    const half
        *__restrict__ value, // [batch_size, num_heads, seq_len, head_dim]
    half
        *__restrict__ output, // [batch_size, num_heads, seq_len, head_dim]
    const QuantParams quant_params, int batch_size, int num_heads,
    int seq_len, int head_dim,
    float scale) // 1/sqrt(head_dim) for scaled dot-product attention
{
  // Block indices
  const int bi        = blockIdx.x; // Batch index
  const int hi        = blockIdx.y; // Head index
  const int row_block = blockIdx.z; // Block of seq_len (query dimension)

  // Thread indices within block
  const int tx  = threadIdx.x;          // Thread x-coordinate
  const int ty  = threadIdx.y;          // Thread y-coordinate
  const int tid = ty * blockDim.x + tx; // Flattened thread index

  // Starting positions
  const int row_start       = row_block * BLOCK_SIZE_M;
  const int row_end         = min(row_start + BLOCK_SIZE_M, seq_len);
  const int rows_this_block = row_end - row_start;

  // Local row index within the block
  const int row_idx = row_start + ty;

  // Shared memory allocations for tiling
  __shared__ __nv_fp8_e4m3
      q_tile[BLOCK_SIZE_M][BLOCK_SIZE_K]; // Query tile in FP8
  __shared__ __nv_fp8_e4m3
      k_tile[BLOCK_SIZE_N][BLOCK_SIZE_K];             // Key tile in FP8
  __shared__ half v_tile[BLOCK_SIZE_N][BLOCK_SIZE_K]; // Value tile in FP16
  __shared__ half
      s_tile[BLOCK_SIZE_M][BLOCK_SIZE_N]; // Attention scores for this tile

  // Buffers for accumulating the output
  half row_output[BLOCK_SIZE_K];
  for (int i = 0; i < BLOCK_SIZE_K; i++) {
    row_output[i] = __float2half(0.0f);
  }

  // Variables for online softmax
  float m_prev = -INFINITY; // Max value seen so far
  float m_curr = -INFINITY; // Max value in current block
  float d_prev = 0.0f;      // Denominator seen so far
  float d_curr = 0.0f;      // Denominator for current block

  // Process key-value blocks in sequence length dimension
  for (int col_block = 0;
       col_block < (seq_len + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
       col_block++) {
    const int col_start       = col_block * BLOCK_SIZE_N;
    const int col_end         = min(col_start + BLOCK_SIZE_N, seq_len);
    const int cols_this_block = col_end - col_start;

    // Clear shared memory tiles
    for (int i = tid; i < BLOCK_SIZE_M * BLOCK_SIZE_N;
         i += blockDim.x * blockDim.y) {
      const int row = i / BLOCK_SIZE_N;
      const int col = i % BLOCK_SIZE_N;
      if (row < rows_this_block && col < cols_this_block) {
        s_tile[row][col] = __float2half(0.0f);
      }
    }
    __syncthreads();

    // Process embedding dimension blocks (k-dimension)
    for (int k_block = 0;
         k_block < (head_dim + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
         k_block++) {
      const int k_start      = k_block * BLOCK_SIZE_K;
      const int k_end        = min(k_start + BLOCK_SIZE_K, head_dim);
      const int k_this_block = k_end - k_start;

      // Load query block into shared memory and quantize to FP8
      for (int i = tid; i < rows_this_block * k_this_block;
           i += blockDim.x * blockDim.y) {
        const int local_row = i / k_this_block;
        const int local_k   = i % k_this_block;

        if (row_start + local_row < seq_len &&
            k_start + local_k < head_dim) {
          // Compute flat index for query
          int q_idx =
              ((bi * num_heads + hi) * seq_len + (row_start + local_row)) *
                  head_dim +
              (k_start + local_k);
          half q_val = query[q_idx];

          // Quantize to FP8_E4M3
          float fp32_val = __half2float(q_val) / quant_params.scale_q;
          fp32_val       = fmaxf(fminf(fp32_val, 448.0f), -448.0f);
          q_tile[local_row][local_k] = __nv_fp8_e4m3(fp32_val);
        }
      }

      // Load key block into shared memory and quantize to FP8
      for (int i = tid; i < cols_this_block * k_this_block;
           i += blockDim.x * blockDim.y) {
        const int local_col = i / k_this_block;
        const int local_k   = i % k_this_block;

        if (col_start + local_col < seq_len &&
            k_start + local_k < head_dim) {
          // Compute flat index for key
          int k_idx =
              ((bi * num_heads + hi) * seq_len + (col_start + local_col)) *
                  head_dim +
              (k_start + local_k);
          half k_val = key[k_idx];

          // Quantize to FP8_E4M3
          float fp32_val = __half2float(k_val) / quant_params.scale_k;
          fp32_val       = fmaxf(fminf(fp32_val, 448.0f), -448.0f);
          k_tile[local_col][local_k] = __nv_fp8_e4m3(fp32_val);
        }
      }

      // Wait for all threads to finish loading Q and K tiles
      __syncthreads();

      // Compute Q * K^T in reduced precision for this block
      if (row_idx < row_end) {
        for (int col_idx = col_start + tx; col_idx < col_end;
             col_idx += blockDim.x) {
          const int local_col = col_idx - col_start;
          float dot_product   = 0.0f;

          // Compute dot product of quantized vectors
          for (int k = 0; k < k_this_block; k++) {
            float q_val =
                (float)q_tile[ty][k]; // Direct conversion instead of
                                      // helper function
            float k_val =
                (float)k_tile[local_col][k]; // Direct conversion instead
                                             // of helper function
            dot_product += q_val * k_val;
          }

          // Apply scaling and dequantization
          dot_product = dot_product * scale * quant_params.scale_qk;

          // Store in shared memory
          s_tile[ty][local_col] = __float2half(dot_product);
        }
      }

      // Wait for all threads to finish computing attention scores
      __syncthreads();
    }

    // Apply online softmax to this block of scores
    if (row_idx < row_end) {
      half row_scores[BLOCK_SIZE_N];
      for (int i = 0; i < cols_this_block; i++) {
        row_scores[i] = s_tile[ty][i];
      }

      // Apply online softmax
      improved_online_softmax_two(row_scores, m_prev, m_curr, d_prev, d_curr,
                     cols_this_block);

      // Update the shared memory with softmax results
      for (int i = 0; i < cols_this_block; i++) {
        s_tile[ty][i] = row_scores[i];
      }
    }
    __syncthreads();

    // Load value block in FP16 precision
    for (int i = tid; i < cols_this_block * head_dim;
         i += blockDim.x * blockDim.y) {
      const int local_col = i / head_dim;
      const int local_k   = i % head_dim;

      if (local_k < head_dim) {
        // Calculate flat index for value
        int v_idx =
            ((bi * num_heads + hi) * seq_len + (col_start + local_col)) *
                head_dim +
            local_k;

        if (local_k < BLOCK_SIZE_K && col_start + local_col < seq_len) {
          v_tile[local_col][local_k] = value[v_idx];
        }
      }
    }
    __syncthreads();

    // Compute attention output in FP16 (softmax(Q*K^T) * V)
    if (row_idx < row_end) {
      for (int k = tx; k < head_dim; k += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < cols_this_block; j++) {
          half attn_weight = s_tile[ty][j];
          half val         = (k < BLOCK_SIZE_K)
                                 ? v_tile[j][k]
                                 : value[((bi * num_heads + hi) * seq_len +
                                  (col_start + j)) *
                                     head_dim +
                                 k];
          acc += __half2float(attn_weight) * __half2float(val);
        }

        // Accumulate in the output buffer
        if (k < BLOCK_SIZE_K) {
          row_output[k] = __float2half(__half2float(row_output[k]) + acc);
        } else {
          int out_idx =
              ((bi * num_heads + hi) * seq_len + row_idx) * head_dim + k;
          output[out_idx] =
              __float2half(__half2float(output[out_idx]) + acc);
        }
      }
    }
    __syncthreads();
  }

  // Write accumulated output for the first BLOCK_SIZE_K elements
  if (row_idx < seq_len) {
    for (int k = tx; k < BLOCK_SIZE_K && k < head_dim; k += blockDim.x) {
      int out_idx =
          ((bi * num_heads + hi) * seq_len + row_idx) * head_dim + k;
      output[out_idx] = row_output[k];
    }
  }
}

/**
 * Main kernel for F32 attention with FlashAttention-style tiling
 * Dimensions:
 * - batch_size: number of sequences in the batch
 * - num_heads: number of attention heads
 * - seq_len: sequence length
 * - head_dim: dimensionality of each attention head
 */
__global__ void fp32_attention_kernel(
    const float
        *__restrict__ query, // [batch_size, num_heads, seq_len, head_dim]
    const float
        *__restrict__ key, // [batch_size, num_heads, seq_len, head_dim]
    const float
        *__restrict__ value, // [batch_size, num_heads, seq_len, head_dim]
    float
        *__restrict__ output, // [batch_size, num_heads, seq_len, head_dim]
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale) // 1/sqrt(head_dim) for scaled dot-product attention
{
  // Block indices
  const int bi        = blockIdx.x; // Batch index
  const int hi        = blockIdx.y; // Head index
  const int row_block = blockIdx.z; // Block of seq_len (query dimension)

  // Thread indices within block
  const int tx  = threadIdx.x;          // Thread x-coordinate
  const int ty  = threadIdx.y;          // Thread y-coordinate
  const int tid = ty * blockDim.x + tx; // Flattened thread index

  // Starting positions
  const int row_start       = row_block * BLOCK_SIZE_M;
  const int row_end         = min(row_start + BLOCK_SIZE_M, seq_len);
  const int rows_this_block = row_end - row_start;

  // Local row index within the block
  const int row_idx = row_start + ty;

  // Shared memory allocations for tiling
  __shared__ float q_tile[BLOCK_SIZE_M][BLOCK_SIZE_K]; // Query tile in FP8
  __shared__ float k_tile[BLOCK_SIZE_N][BLOCK_SIZE_K]; // Key tile in FP8
  __shared__ float v_tile[BLOCK_SIZE_N]
                         [BLOCK_SIZE_K]; // Value tile in FP16
  __shared__ float s_tile[BLOCK_SIZE_M]
                         [BLOCK_SIZE_N]; // Attention scores for this tile

  // Buffers for accumulating the output
  float row_output[BLOCK_SIZE_K];
  for (int i = 0; i < BLOCK_SIZE_K; i++) {
    row_output[i] = 0.0f;
  }

  // Variables for online softmax
  float m_prev = -INFINITY; // Max value seen so far
  float m_curr = -INFINITY; // Max value in current block
  float d_prev = 0.0f;      // Denominator seen so far
  float d_curr = 0.0f;      // Denominator for current block

  // Process key-value blocks in sequence length dimension
  for (int col_block = 0;
       col_block < (seq_len + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
       col_block++) {
    const int col_start       = col_block * BLOCK_SIZE_N;
    const int col_end         = min(col_start + BLOCK_SIZE_N, seq_len);
    const int cols_this_block = col_end - col_start;

    // Clear shared memory tiles
    for (int i = tid; i < BLOCK_SIZE_M * BLOCK_SIZE_N;
         i += blockDim.x * blockDim.y) {
      const int row = i / BLOCK_SIZE_N;
      const int col = i % BLOCK_SIZE_N;
      if (row < rows_this_block && col < cols_this_block) {
        s_tile[row][col] = 0.0f;
      }
    }
    __syncthreads();

    // Process embedding dimension blocks (k-dimension)
    for (int k_block = 0;
         k_block < (head_dim + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
         k_block++) {
      const int k_start      = k_block * BLOCK_SIZE_K;
      const int k_end        = min(k_start + BLOCK_SIZE_K, head_dim);
      const int k_this_block = k_end - k_start;

      // Load query block into shared memory and
      for (int i = tid; i < rows_this_block * k_this_block;
           i += blockDim.x * blockDim.y) {
        const int local_row = i / k_this_block;
        const int local_k   = i % k_this_block;

        if (row_start + local_row < seq_len &&
            k_start + local_k < head_dim) {
          // Compute flat index for query
          int q_idx =
              ((bi * num_heads + hi) * seq_len + (row_start + local_row)) *
                  head_dim +
              (k_start + local_k);
          float q_val                = query[q_idx];
          q_tile[local_row][local_k] = q_val;
        }
      }

      // Load key block into shared memory and
      for (int i = tid; i < cols_this_block * k_this_block;
           i += blockDim.x * blockDim.y) {
        const int local_col = i / k_this_block;
        const int local_k   = i % k_this_block;

        if (col_start + local_col < seq_len &&
            k_start + local_k < head_dim) {
          // Compute flat index for key
          int k_idx =
              ((bi * num_heads + hi) * seq_len + (col_start + local_col)) *
                  head_dim +
              (k_start + local_k);
          float k_val                = key[k_idx];
          k_tile[local_col][local_k] = k_val;
        }
      }

      // Wait for all threads to finish loading Q and K tiles
      __syncthreads();

      // Compute Q * K^T in reduced precision for this block
      if (row_idx < row_end) {
        for (int col_idx = col_start + tx; col_idx < col_end;
             col_idx += blockDim.x) {
          const int local_col = col_idx - col_start;
          float dot_product   = 0.0f;

          // Compute dot product of quantized vectors
          for (int k = 0; k < k_this_block; k++) {
            float q_val =
                (float)q_tile[ty][k]; // Direct conversion instead of
                                      // helper function
            float k_val =
                (float)k_tile[local_col][k]; // Direct conversion instead
                                             // of helper function
            dot_product += q_val * k_val;
          }

          // Store in shared memory
          s_tile[ty][local_col] = dot_product;
        }
      }

      // Wait for all threads to finish computing attention scores
      __syncthreads();
    }

    // Apply online softmax to this block of scores
    if (row_idx < row_end) {
      float row_scores[BLOCK_SIZE_N];
      for (int i = 0; i < cols_this_block; i++) {
        row_scores[i] = s_tile[ty][i];
      }

      // Apply online softmax
      online_softmax_full(row_scores, m_prev, m_curr, d_prev, d_curr,
                          cols_this_block);

      // Update the shared memory with softmax results
      for (int i = 0; i < cols_this_block; i++) {
        s_tile[ty][i] = row_scores[i];
      }
    }
    __syncthreads();

    // Load value block in FP16 precision
    for (int i = tid; i < cols_this_block * head_dim;
         i += blockDim.x * blockDim.y) {
      const int local_col = i / head_dim;
      const int local_k   = i % head_dim;

      if (local_k < head_dim) {
        // Calculate flat index for value
        int v_idx =
            ((bi * num_heads + hi) * seq_len + (col_start + local_col)) *
                head_dim +
            local_k;

        if (local_k < BLOCK_SIZE_K && col_start + local_col < seq_len) {
          v_tile[local_col][local_k] = value[v_idx];
        }
      }
    }
    __syncthreads();

    // Compute attention output
    if (row_idx < row_end) {
      for (int k = tx; k < head_dim; k += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < cols_this_block; j++) {
          float attn_weight = s_tile[ty][j];
          float val         = (k < BLOCK_SIZE_K)
                                  ? v_tile[j][k]
                                  : value[((bi * num_heads + hi) * seq_len +
                                   (col_start + j)) *
                                      head_dim +
                                  k];
          acc += attn_weight * val;
        }

        // Accumulate in the output buffer
        if (k < BLOCK_SIZE_K) {
          row_output[k] = (row_output[k] + acc);
        } else {
          int out_idx =
              ((bi * num_heads + hi) * seq_len + row_idx) * head_dim + k;
          output[out_idx] = (output[out_idx] + acc);
        }
      }
    }
    __syncthreads();
  }

  // Write accumulated output for the first BLOCK_SIZE_K elements
  if (row_idx < seq_len) {
    for (int k = tx; k < BLOCK_SIZE_K && k < head_dim; k += blockDim.x) {
      int out_idx =
          ((bi * num_heads + hi) * seq_len + row_idx) * head_dim + k;
      output[out_idx] = row_output[k];
    }
  }
}

// ----------------------------------------------------------------------------
// 2) NAÏVE FP32 ATTENTION KERNEL (no tiling, one thread per output
// element)
// ----------------------------------------------------------------------------
__global__ void
fp32_naive_attention_kernel(const float *__restrict__ query, // [B,H,L,D]
                            const float *__restrict__ key,   // [B,H,L,D]
                            const float *__restrict__ value, // [B,H,L,D]
                            float *__restrict__ output,      // [B,H,L,D]
                            int B, int H, int L, int D) {
  // Flatten grid:  blockIdx.x = (b*H + h), blockIdx.y = q_pos, threadIdx.x
  // = head_dim index
  int bh = blockIdx.x, pos = blockIdx.y;
  int d = threadIdx.x;
  int b = bh / H, h = bh % H;

  // Pointers to this head
  const float *Q = query + (bh * L * D);
  const float *K = key + (bh * L * D);
  const float *V = value + (bh * L * D);
  float *O       = output + (bh * L * D);

  // 1) compute all scores for this query pos
  extern __shared__ float
      scratch[]; // size = 2*L: [0..L) = scores, [L..2L) = exps
  float *S = scratch;
  float *E = scratch + L;
  float m  = -1e30f;

  // dot Q_pos · K_j for j=0..L-1
  for (int j = 0; j < L; j++) {
    float dot = 0;
    for (int x = 0; x < D; x++) {
      dot += Q[pos * D + x] * K[j * D + x];
    }
    S[j] = dot / sqrtf(float(D));
    m    = fmaxf(m, S[j]);
  }
  // softmax
  float sum = 0;
  for (int j = 0; j < L; j++) {
    float e = expf(S[j] - m);
    E[j]    = e;
    sum += e;
  }
  // 2) weighted sum over V
  float acc = 0;
  for (int j = 0; j < L; j++) {
    acc += (E[j] / (sum + 1e-6f)) * V[j * D + d];
  }
  // write output element
  O[pos * D + d] = acc;
}

// Helper function to determine quantization parameters
void compute_quant_params(const half *query, const half *key,
                          int batch_size, int num_heads, int seq_len,
                          int head_dim, QuantParams &params) {

  params.scale_q = 10.0f; // the scale should be determined by data range
                          // of the actuall llm embedding
  params.scale_k  = 10.0f;
  params.scale_qk = params.scale_q * params.scale_k;
}

/**
 * Kernel to find min/max values in parallel
 */
__global__ void findMinMax(const half* data, float* min_max_values, int size) {
    extern __shared__ float shared_data[];
    float* shared_min = shared_data;
    float* shared_max = shared_data + blockDim.x;
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize local min/max
    float thread_min = FLT_MAX;
    float thread_max = -FLT_MAX;
    
    // Process multiple elements per thread
    for (int i = gid; i < size; i += blockDim.x * gridDim.x) {
        if (i < size) {
            float val = __half2float(data[i]);
            thread_min = fminf(thread_min, val);
            thread_max = fmaxf(thread_max, val);
        }
    }
    
    // Initialize shared memory
    shared_min[tid] = thread_min;
    shared_max[tid] = thread_max;
    __syncthreads();
    
    // Parallel reduction to find min/max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_min[tid] = fminf(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    
    // Write results to global memory
    if (tid == 0) {
        min_max_values[blockIdx.x * 2] = shared_min[0];
        min_max_values[blockIdx.x * 2 + 1] = shared_max[0];
    }
}

/**
 * Improved function to compute quantization parameters
 * Uses per-head quantization and adaptive scaling techniques.
 */
void compute_quant_params_two(const half *query, const half *key,
                          int batch_size, int num_heads, int seq_len,
                          int head_dim, QuantParams &params) {
    // Constants for FP8 E4M3 format
    const float fp8_max_value = 448.0f;
    const float safety_factor = 0.98f;  // Increased from 0.9 to 0.98
    
    // If test data shows better results with fixed scales, use a hybrid approach
    const bool use_fixed_scale = false;  // Set based on empirical tests
    const float fixed_scale_value = 10.0f;
    
    // For per-head or global quantization
    const bool use_per_head = false;  // Toggle for per-head quantization
    
    if (use_fixed_scale) {
        // Use the original fixed scale that worked well in your tests
        params.scale_q = fixed_scale_value;
        params.scale_k = fixed_scale_value;
        params.scale_qk = params.scale_q * params.scale_k;
        return;
    }
    
    // Allocate device memory for min/max values
    const int num_blocks = 256;
    float *d_q_min_max, *d_k_min_max;
    cudaMalloc(&d_q_min_max, num_blocks * 2 * sizeof(float));
    cudaMalloc(&d_k_min_max, num_blocks * 2 * sizeof(float));
    
    // Configuration for kernel launch
    int threads_per_block = 256;
    int shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    if (use_per_head) {
        // Per-head quantization - calculate scales for each attention head
        params.scale_q = 0;  // Will compute average scale across heads
        params.scale_k = 0;
        
        for (int h = 0; h < num_heads; h++) {
            // Calculate offsets for this head
            size_t head_elements = seq_len * head_dim;
            const half* q_head = query + h * head_elements;
            const half* k_head = key + h * head_elements;
            
            // Find min/max for this head
            findMinMax<<<num_blocks, threads_per_block, shared_mem_size>>>(
                q_head, d_q_min_max, head_elements);
            findMinMax<<<num_blocks, threads_per_block, shared_mem_size>>>(
                k_head, d_k_min_max, head_elements);
            
            // Copy results back to host
            float *h_q_min_max = new float[num_blocks * 2];
            float *h_k_min_max = new float[num_blocks * 2];
            
            cudaMemcpy(h_q_min_max, d_q_min_max, num_blocks * 2 * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_k_min_max, d_k_min_max, num_blocks * 2 * sizeof(float), cudaMemcpyDeviceToHost);
            
            // Find global min/max from block results
            float q_min = FLT_MAX, q_max = -FLT_MAX;
            float k_min = FLT_MAX, k_max = -FLT_MAX;
            
            for (int i = 0; i < num_blocks; i++) {
                q_min = fminf(q_min, h_q_min_max[i * 2]);
                q_max = fmaxf(q_max, h_q_min_max[i * 2 + 1]);
                k_min = fminf(k_min, h_k_min_max[i * 2]);
                k_max = fmaxf(k_max, h_k_min_max[i * 2 + 1]);
            }
            
            // Compute absolute max values
            float q_abs_max = fmaxf(fabsf(q_min), fabsf(q_max));
            float k_abs_max = fmaxf(fabsf(k_min), fabsf(k_max));
            
            // Add to running total for average calculation
            float head_scale_q = q_abs_max > 0 ? (fp8_max_value * safety_factor) / q_abs_max : 1.0f;
            float head_scale_k = k_abs_max > 0 ? (fp8_max_value * safety_factor) / k_abs_max : 1.0f;
            
            params.scale_q += head_scale_q;
            params.scale_k += head_scale_k;
            
            delete[] h_q_min_max;
            delete[] h_k_min_max;
        }
        
        // Compute average scale across heads
        params.scale_q /= num_heads;
        params.scale_k /= num_heads;
        
    } else {
        // Global quantization (across all heads)
        int total_elements = batch_size * num_heads * seq_len * head_dim;
        
        // Launch kernel to find min/max
        findMinMax<<<num_blocks, threads_per_block, shared_mem_size>>>(
            query, d_q_min_max, total_elements);
        findMinMax<<<num_blocks, threads_per_block, shared_mem_size>>>(
            key, d_k_min_max, total_elements);
        
        // Copy results back to host
        float *h_q_min_max = new float[num_blocks * 2];
        float *h_k_min_max = new float[num_blocks * 2];
        
        cudaMemcpy(h_q_min_max, d_q_min_max, num_blocks * 2 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_k_min_max, d_k_min_max, num_blocks * 2 * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Find global min/max from block results
        float q_min = FLT_MAX, q_max = -FLT_MAX;
        float k_min = FLT_MAX, k_max = -FLT_MAX;
        
        for (int i = 0; i < num_blocks; i++) {
            q_min = fminf(q_min, h_q_min_max[i * 2]);
            q_max = fmaxf(q_max, h_q_min_max[i * 2 + 1]);
            k_min = fminf(k_min, h_k_min_max[i * 2]);
            k_max = fmaxf(k_max, h_k_min_max[i * 2 + 1]);
        }
        
        // Compute absolute max values
        float q_abs_max = fmaxf(fabsf(q_min), fabsf(q_max));
        float k_abs_max = fmaxf(fabsf(k_min), fabsf(k_max));
        
        // Hybrid approach - if values are small, fixed scales might work better
        if (q_abs_max < 5.0f && k_abs_max < 5.0f) {
            params.scale_q = fixed_scale_value;
            params.scale_k = fixed_scale_value;
        } else {
            // Calculate scale factors with improved safety factor
            params.scale_q = q_abs_max > 0 ? (fp8_max_value * safety_factor) / q_abs_max : 1.0f;
            params.scale_k = k_abs_max > 0 ? (fp8_max_value * safety_factor) / k_abs_max : 1.0f;
        }
        
        delete[] h_q_min_max;
        delete[] h_k_min_max;
    }
    
    // Set the combined scale
    params.scale_qk = params.scale_q * params.scale_k;
    
    // Clean up device memory
    cudaFree(d_q_min_max);
    cudaFree(d_k_min_max);
    
    // Debug output
    // printf("Scale factors: q=%.4f, k=%.4f, qk=%.4f\n", 
    //    params.scale_q, params.scale_k, params.scale_qk);
}

// Host function to launch the quantized attention kernel
void fp8_quantized_attention(const half *query, const half *key,
                             const half *value, half *output,
                             int batch_size, int num_heads, int seq_len,
                             int head_dim) {
  // Calculate total size
  size_t total_elements = batch_size * num_heads * seq_len * head_dim;

  // Timing variables
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Determine quantization parameters
  cudaEventRecord(start);
  QuantParams quant_params;
  compute_quant_params_two(query, key, batch_size, num_heads, seq_len,
                     head_dim, quant_params);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Computing quant params time: " << milliseconds << " ms" << std::endl;

  // Clear output
  cudaMemset(output, 0, total_elements * sizeof(half));

  // Calculate grid and block dimensions
  dim3 grid(batch_size, num_heads,
            (seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
  dim3 block(32, 4); // 128 threads per block

  // Calculate attention scaling factor
  float scale = 1.0f / sqrtf(head_dim);

  // Launch kernel
  cudaEventRecord(start);
  fp8_flash_attention_kernel<<<grid, block>>>(
      query, key, value, output, quant_params, batch_size, num_heads,
      seq_len, head_dim, scale);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  // std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

  // Check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
  }
}

// Reference CPU implementation of attention for comparison
void reference_attention(const float *query, const float *key,
                         const float *value, float *output, int batch_size,
                         int num_heads, int seq_len, int head_dim) {
  // Scale factor for attention
  float scale = 1.0f / sqrtf(head_dim);

  // For each batch and head
  for (int b = 0; b < batch_size; b++) {
    for (int h = 0; h < num_heads; h++) {
      // Calculate base indices for this batch and head
      int batch_head_offset = (b * num_heads + h) * seq_len * head_dim;

      // First, compute full attention matrix (seq_len x seq_len)
      std::vector<float> attention_scores(seq_len * seq_len, 0.0f);

      // Calculate Q * K^T
      for (int q_seq = 0; q_seq < seq_len; q_seq++) {
        for (int k_seq = 0; k_seq < seq_len; k_seq++) {
          float dot_product = 0.0f;

          // Compute dot product between query and key vectors
          for (int d = 0; d < head_dim; d++) {
            int q_idx = batch_head_offset + q_seq * head_dim + d;
            int k_idx = batch_head_offset + k_seq * head_dim + d;

            float q_val = query[q_idx];
            float k_val = key[k_idx];

            dot_product += q_val * k_val;
          }

          // Apply scaling
          attention_scores[q_seq * seq_len + k_seq] = dot_product * scale;
        }

        // Apply softmax row-wise (for each query sequence position)
        // First find the maximum value for numerical stability
        float max_val = -INFINITY;
        for (int k_seq = 0; k_seq < seq_len; k_seq++) {
          max_val =
              fmaxf(max_val, attention_scores[q_seq * seq_len + k_seq]);
        }

        // Compute exponentials and sum
        float exp_sum = 0.0f;
        for (int k_seq = 0; k_seq < seq_len; k_seq++) {
          float exp_val =
              expf(attention_scores[q_seq * seq_len + k_seq] - max_val);
          attention_scores[q_seq * seq_len + k_seq] = exp_val;
          exp_sum += exp_val;
        }

        // Normalize
        for (int k_seq = 0; k_seq < seq_len; k_seq++) {
          attention_scores[q_seq * seq_len + k_seq] /= exp_sum;
        }

        // Compute weighted sum with values
        for (int d = 0; d < head_dim; d++) {
          float weighted_sum = 0.0f;

          for (int k_seq = 0; k_seq < seq_len; k_seq++) {
            int v_idx         = batch_head_offset + k_seq * head_dim + d;
            float v_val       = value[v_idx];
            float attn_weight = attention_scores[q_seq * seq_len + k_seq];

            weighted_sum += attn_weight * v_val;
          }

          // Store the result
          int out_idx     = batch_head_offset + q_seq * head_dim + d;
          output[out_idx] = weighted_sum;
        }
      }
    }
  }
}
// Reference CPU implementation of attention for comparison
void reference_gpu_attention(const float *query, const float *key,
                             const float *value, float *output,
                             int batch_size, int num_heads, int seq_len,
                             int head_dim, attention_strategy stra) {
  // Scale factor for attention
  float scale = 1.0f / sqrtf(head_dim);

  // Calculate total size
  size_t total_elements = batch_size * num_heads * seq_len * head_dim;

  // Clear output
  cudaMemset(output, 0, total_elements * sizeof(float));

  // Calculate grid and block dimensions
  dim3 grid(batch_size, num_heads,
            (seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
  dim3 block(32, 4); // 128 threads per block

  //   // Launch kernel
  //   switch (stra) {

  //     case attention_strategy::tiled:
  //       fp32_attention_kernel<<<grid, block>>>(query, key, value,
  //       output,
  //                                              batch_size, num_heads,
  //                                              seq_len, head_dim,
  //                                              scale);
  //       break;
  //     case attention_strategy::untiled:
  //       fp32_naive_attention_kernel<<<grid, block>>>(
  //           query, key, value, output, batch_size, num_heads, seq_len,
  //           head_dim, scale);
  //   }

  // Check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
  }
}
