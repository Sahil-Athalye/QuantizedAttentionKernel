#include <cuda_runtime.h>
#include <cuda_fp16.h>  // Added for half-precision support
#include <cmath>
#include <vector>
#include <iostream>

// Define block size parameters - tuned for specific hardware
#define BLOCK_SIZE_M 64  // Tile size for batch and head dimensions
#define BLOCK_SIZE_N 64  // Tile size for sequence length
#define BLOCK_SIZE_K 32  // Tile size for embedding dimension
#define WARP_SIZE 32

// Custom FP8 E4M3 implementation since we can't use native types
struct fp8_e4m3 {
    uint8_t bits;
    
    // Default constructor
    __host__ __device__ fp8_e4m3() : bits(0) {}
    
    // Constructor from float
    __host__ __device__ fp8_e4m3(float val) {
        // E4M3 has 1 sign bit, 4 exponent bits, 3 mantissa bits
        // Range is approximately [-448, 448] with ~0.01 precision near zero
        
        // Handle zero case
        if (val == 0.0f) {
            bits = 0;
            return;
        }
        
        // Extract sign
        uint8_t sign = (val < 0) ? 0x80 : 0;
        val = fabsf(val);
        
        // Get exponent and mantissa
        int exp = (int)floorf(log2f(val));
        float mantissa = val / powf(2.0f, exp) - 1.0f;
        
        // Adjust for E4M3 bounds
        if (exp < -7) {
            // Underflow to zero
            bits = 0;
            return;
        }
        if (exp > 8) {
            // Overflow to max value
            bits = sign | 0x7F;
            return;
        }
        
        // Encode
        uint8_t exp_bits = (exp + 7) & 0xF;
        uint8_t mantissa_bits = (uint8_t)(mantissa * 8.0f);
        
        bits = sign | (exp_bits << 3) | mantissa_bits;
    }
    
    // Conversion to float
    __host__ __device__ operator float() const {
        if ((bits & 0x7F) == 0) return 0.0f; // Handle zero
        
        float sign = (bits & 0x80) ? -1.0f : 1.0f;
        int exp = ((bits >> 3) & 0xF) - 7;
        float mant = 1.0f + (float)(bits & 0x7) / 8.0f;
        
        return sign * mant * powf(2.0f, exp);
    }
};

// Quantization parameters struct - could be dynamically determined
struct QuantParams {
    float scale_q;  // Scale factor for query
    float scale_k;  // Scale factor for key
    float scale_qk; // Combined scale for Q*K
};

// Helper function to quantize from FP16 to FP8_E4M3
__device__ void quantize_fp16_to_fp8(const half* input, fp8_e4m3* output, 
                                    float scale, int size) {
    for (int i = 0; i < size; i++) {
        float fp32_val = __half2float(input[i]);
        // Scale and clamp values to FP8 E4M3 range [-448, 448]
        fp32_val = fp32_val / scale;
        fp32_val = fmaxf(fminf(fp32_val, 448.0f), -448.0f);
        output[i] = fp8_e4m3(fp32_val);
    }
}

// Helper function to dequantize from FP8_E4M3 to FP16
__device__ void dequantize_fp8_to_fp16(const fp8_e4m3* input, half* output, 
                                      float scale, int size) {
    for (int i = 0; i < size; i++) {
        // Convert to FP32 then FP16
        float fp32_val = (float)input[i];
        fp32_val = fp32_val * scale;
        output[i] = __float2half(fp32_val);
    }
}

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

/**
 * Main kernel for FP8 quantized attention with FlashAttention-style tiling
 * Dimensions:
 * - batch_size: number of sequences in the batch
 * - num_heads: number of attention heads
 * - seq_len: sequence length
 * - head_dim: dimensionality of each attention head
 */
__global__ void fp8_flash_attention_kernel(
    const half* __restrict__ query,        // [batch_size, num_heads, seq_len, head_dim]
    const half* __restrict__ key,          // [batch_size, num_heads, seq_len, head_dim]
    const half* __restrict__ value,        // [batch_size, num_heads, seq_len, head_dim]
    half* __restrict__ output,             // [batch_size, num_heads, seq_len, head_dim]
    const QuantParams quant_params,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale)                           // 1/sqrt(head_dim) for scaled dot-product attention
{
    // Block indices
    const int bi = blockIdx.x;            // Batch index
    const int hi = blockIdx.y;            // Head index
    const int row_block = blockIdx.z;     // Block of seq_len (query dimension)
    
    // Thread indices within block
    const int tx = threadIdx.x;           // Thread x-coordinate
    const int ty = threadIdx.y;           // Thread y-coordinate
    const int tid = ty * blockDim.x + tx; // Flattened thread index
    
    // Starting positions
    const int row_start = row_block * BLOCK_SIZE_M;
    const int row_end = min(row_start + BLOCK_SIZE_M, seq_len);
    const int rows_this_block = row_end - row_start;
    
    // Local row index within the block
    const int row_idx = row_start + ty;
    
    // Shared memory allocations for tiling
    __shared__ fp8_e4m3 q_tile[BLOCK_SIZE_M][BLOCK_SIZE_K];  // Query tile in FP8
    __shared__ fp8_e4m3 k_tile[BLOCK_SIZE_N][BLOCK_SIZE_K];  // Key tile in FP8
    __shared__ half v_tile[BLOCK_SIZE_N][BLOCK_SIZE_K];      // Value tile in FP16
    __shared__ half s_tile[BLOCK_SIZE_M][BLOCK_SIZE_N];      // Attention scores for this tile
    
    // Buffers for accumulating the output
    half row_output[BLOCK_SIZE_K];
    for (int i = 0; i < BLOCK_SIZE_K; i++) {
        row_output[i] = __float2half(0.0f);
    }
    
    // Variables for online softmax
    float m_prev = -INFINITY;  // Max value seen so far
    float m_curr = -INFINITY;  // Max value in current block
    float d_prev = 0.0f;       // Denominator seen so far
    float d_curr = 0.0f;       // Denominator for current block
    
    // Process key-value blocks in sequence length dimension
    for (int col_block = 0; col_block < (seq_len + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N; col_block++) {
        const int col_start = col_block * BLOCK_SIZE_N;
        const int col_end = min(col_start + BLOCK_SIZE_N, seq_len);
        const int cols_this_block = col_end - col_start;
        
        // Clear shared memory tiles
        for (int i = tid; i < BLOCK_SIZE_M * BLOCK_SIZE_N; i += blockDim.x * blockDim.y) {
            const int row = i / BLOCK_SIZE_N;
            const int col = i % BLOCK_SIZE_N;
            if (row < rows_this_block && col < cols_this_block) {
                s_tile[row][col] = __float2half(0.0f);
            }
        }
        __syncthreads();
        
        // Process embedding dimension blocks (k-dimension)
        for (int k_block = 0; k_block < (head_dim + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; k_block++) {
            const int k_start = k_block * BLOCK_SIZE_K;
            const int k_end = min(k_start + BLOCK_SIZE_K, head_dim);
            const int k_this_block = k_end - k_start;
            
            // Load query block into shared memory and quantize to FP8
            for (int i = tid; i < rows_this_block * k_this_block; i += blockDim.x * blockDim.y) {
                const int local_row = i / k_this_block;
                const int local_k = i % k_this_block;
                
                if (row_start + local_row < seq_len && k_start + local_k < head_dim) {
                    // Compute flat index for query
                    int q_idx = ((bi * num_heads + hi) * seq_len + (row_start + local_row)) * head_dim + (k_start + local_k);
                    half q_val = query[q_idx];
                    
                    // Quantize to FP8_E4M3
                    float fp32_val = __half2float(q_val) / quant_params.scale_q;
                    fp32_val = fmaxf(fminf(fp32_val, 448.0f), -448.0f);
                    q_tile[local_row][local_k] = fp8_e4m3(fp32_val);
                }
            }
            
            // Load key block into shared memory and quantize to FP8
            for (int i = tid; i < cols_this_block * k_this_block; i += blockDim.x * blockDim.y) {
                const int local_col = i / k_this_block;
                const int local_k = i % k_this_block;
                
                if (col_start + local_col < seq_len && k_start + local_k < head_dim) {
                    // Compute flat index for key
                    int k_idx = ((bi * num_heads + hi) * seq_len + (col_start + local_col)) * head_dim + (k_start + local_k);
                    half k_val = key[k_idx];
                    
                    // Quantize to FP8_E4M3
                    float fp32_val = __half2float(k_val) / quant_params.scale_k;
                    fp32_val = fmaxf(fminf(fp32_val, 448.0f), -448.0f);
                    k_tile[local_col][local_k] = fp8_e4m3(fp32_val);
                }
            }
            
            // Wait for all threads to finish loading Q and K tiles
            __syncthreads();
            
            // Compute Q * K^T in reduced precision for this block
            if (row_idx < row_end) {
                for (int col_idx = col_start + tx; col_idx < col_end; col_idx += blockDim.x) {
                    const int local_col = col_idx - col_start;
                    float dot_product = 0.0f;
                    
                    // Compute dot product of quantized vectors
                    for (int k = 0; k < k_this_block; k++) {
                        float q_val = (float)q_tile[ty][k]; // Direct conversion instead of helper function
                        float k_val = (float)k_tile[local_col][k]; // Direct conversion instead of helper function
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
            online_softmax(row_scores, m_prev, m_curr, d_prev, d_curr, cols_this_block);
            
            // Update the shared memory with softmax results
            for (int i = 0; i < cols_this_block; i++) {
                s_tile[ty][i] = row_scores[i];
            }
        }
        __syncthreads();
        
        // Load value block in FP16 precision
        for (int i = tid; i < cols_this_block * head_dim; i += blockDim.x * blockDim.y) {
            const int local_col = i / head_dim;
            const int local_k = i % head_dim;
            
            if (local_k < head_dim) {
                // Calculate flat index for value
                int v_idx = ((bi * num_heads + hi) * seq_len + (col_start + local_col)) * head_dim + local_k;
                
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
                    half val = (k < BLOCK_SIZE_K) ? v_tile[j][k] : 
                        value[((bi * num_heads + hi) * seq_len + (col_start + j)) * head_dim + k];
                    acc += __half2float(attn_weight) * __half2float(val);
                }
                
                // Accumulate in the output buffer
                if (k < BLOCK_SIZE_K) {
                    row_output[k] = __float2half(__half2float(row_output[k]) + acc);
                } else {
                    int out_idx = ((bi * num_heads + hi) * seq_len + row_idx) * head_dim + k;
                    output[out_idx] = __float2half(__half2float(output[out_idx]) + acc);
                }
            }
        }
        __syncthreads();
    }
    
    // Write accumulated output for the first BLOCK_SIZE_K elements
    if (row_idx < seq_len) {
        for (int k = tx; k < BLOCK_SIZE_K && k < head_dim; k += blockDim.x) {
            int out_idx = ((bi * num_heads + hi) * seq_len + row_idx) * head_dim + k;
            output[out_idx] = row_output[k];
        }
    }
}

// Helper function to determine quantization parameters
void compute_quant_params(const half* query, const half* key, 
                         int batch_size, int num_heads, int seq_len, int head_dim,
                         QuantParams& params) {

    params.scale_q = 10.0f;  // the scale should be determined by data range of the actuall llm embedding
    params.scale_k = 10.0f;  
    params.scale_qk = params.scale_q * params.scale_k;
}

// Host function to launch the quantized attention kernel
void fp8_quantized_attention(const half* query, const half* key, const half* value, 
                           half* output, int batch_size, int num_heads, 
                           int seq_len, int head_dim) {
    // Calculate total size
    size_t total_elements = batch_size * num_heads * seq_len * head_dim;
    
    // Determine quantization parameters
    QuantParams quant_params;
    compute_quant_params(query, key, batch_size, num_heads, seq_len, head_dim, quant_params);
    
    // Clear output
    cudaMemset(output, 0, total_elements * sizeof(half));
    
    // Calculate grid and block dimensions
    dim3 grid(batch_size, num_heads, (seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    dim3 block(32, 4); // 128 threads per block
    
    // Calculate attention scaling factor
    float scale = 1.0f / sqrtf(head_dim);
    
    // Launch kernel
    fp8_flash_attention_kernel<<<grid, block>>>(
        query, key, value, output, quant_params,
        batch_size, num_heads, seq_len, head_dim, scale
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
}

// Reference CPU implementation of attention for comparison
void reference_attention(const half* query, const half* key, const half* value, 
    half* output, int batch_size, int num_heads, 
    int seq_len, int head_dim) {
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
     
     float q_val = __half2float(query[q_idx]);
     float k_val = __half2float(key[k_idx]);
     
     dot_product += q_val * k_val;
 }
 
 // Apply scaling
 attention_scores[q_seq * seq_len + k_seq] = dot_product * scale;
}

// Apply softmax row-wise (for each query sequence position)
// First find the maximum value for numerical stability
float max_val = -INFINITY;
for (int k_seq = 0; k_seq < seq_len; k_seq++) {
 max_val = fmaxf(max_val, attention_scores[q_seq * seq_len + k_seq]);
}

// Compute exponentials and sum
float exp_sum = 0.0f;
for (int k_seq = 0; k_seq < seq_len; k_seq++) {
 float exp_val = expf(attention_scores[q_seq * seq_len + k_seq] - max_val);
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
     int v_idx = batch_head_offset + k_seq * head_dim + d;
     float v_val = __half2float(value[v_idx]);
     float attn_weight = attention_scores[q_seq * seq_len + k_seq];
     
     weighted_sum += attn_weight * v_val;
 }
 
 // Store the result
 int out_idx = batch_head_offset + q_seq * head_dim + d;
 output[out_idx] = __float2half(weighted_sum);
}
}
}
}
}

// Complete test function that uses the reference implementation 
// to validate the GPU kernel
void test_fp8_flash_attention() {
// Small test configuration
int batch_size = 1;
int num_heads = 1;
int seq_len = 8;
int head_dim = 8;

std::cout << "Testing FP8 Flash Attention with dimensions: " 
<< "batch_size=" << batch_size 
<< ", num_heads=" << num_heads 
<< ", seq_len=" << seq_len 
<< ", head_dim=" << head_dim << std::endl;

// Allocate host memory
std::vector<half> h_query(batch_size * num_heads * seq_len * head_dim);
std::vector<half> h_key(batch_size * num_heads * seq_len * head_dim);
std::vector<half> h_value(batch_size * num_heads * seq_len * head_dim);
std::vector<half> h_output_gpu(batch_size * num_heads * seq_len * head_dim);
std::vector<half> h_output_cpu(batch_size * num_heads * seq_len * head_dim);

// Initialize with simple predictable pattern
for (int i = 0; i < batch_size * num_heads * seq_len * head_dim; i++) {
// Initialize query with identity pattern
int seq_idx = (i / head_dim) % seq_len;
int dim_idx = i % head_dim;

if (seq_idx == dim_idx) {
h_query[i] = __float2half(1.0f);
} else {
h_query[i] = __float2half(0.0f);
}

// Initialize key the same as query for this test
h_key[i] = h_query[i];

// Initialize value with small increasing values
h_value[i] = __float2half(0.1f * (i % 10));
}

// Allocate device memory
half *d_query, *d_key, *d_value, *d_output;
size_t size = batch_size * num_heads * seq_len * head_dim * sizeof(half);

cudaError_t error;
error = cudaMalloc(&d_query, size);
if (error != cudaSuccess) {
std::cerr << "cudaMalloc failed: " << cudaGetErrorString(error) << std::endl;
return;
}

error = cudaMalloc(&d_key, size);
if (error != cudaSuccess) {
std::cerr << "cudaMalloc failed: " << cudaGetErrorString(error) << std::endl;
cudaFree(d_query);
return;
}

error = cudaMalloc(&d_value, size);
if (error != cudaSuccess) {
std::cerr << "cudaMalloc failed: " << cudaGetErrorString(error) << std::endl;
cudaFree(d_query);
cudaFree(d_key);
return;
}

error = cudaMalloc(&d_output, size);
if (error != cudaSuccess) {
std::cerr << "cudaMalloc failed: " << cudaGetErrorString(error) << std::endl;
cudaFree(d_query);
cudaFree(d_key);
cudaFree(d_value);
return;
}

// Copy input data to device
cudaMemcpy(d_query, h_query.data(), size, cudaMemcpyHostToDevice);
cudaMemcpy(d_key, h_key.data(), size, cudaMemcpyHostToDevice);
cudaMemcpy(d_value, h_value.data(), size, cudaMemcpyHostToDevice);

// Clear output memory
cudaMemset(d_output, 0, size);

// Run GPU implementation
std::cout << "Running GPU implementation..." << std::endl;
fp8_quantized_attention(d_query, d_key, d_value, d_output, 
        batch_size, num_heads, seq_len, head_dim);

// Check for kernel launch errors
error = cudaGetLastError();
if (error != cudaSuccess) {
std::cerr << "Kernel launch error: " << cudaGetErrorString(error) << std::endl;
}

// Wait for kernel to finish
error = cudaDeviceSynchronize();
if (error != cudaSuccess) {
std::cerr << "Kernel execution error: " << cudaGetErrorString(error) << std::endl;
}

// Copy result back
cudaMemcpy(h_output_gpu.data(), d_output, size, cudaMemcpyDeviceToHost);

// Run reference CPU implementation
std::cout << "Running CPU reference implementation..." << std::endl;
reference_attention(h_query.data(), h_key.data(), h_value.data(), 
    h_output_cpu.data(), batch_size, num_heads, seq_len, head_dim);

// Compare results
float max_abs_error = 0.0f;
float max_rel_error = 0.0f;
int max_error_idx = -1;

for (int i = 0; i < batch_size * num_heads * seq_len * head_dim; i++) {
float gpu_val = __half2float(h_output_gpu[i]);
float cpu_val = __half2float(h_output_cpu[i]);
float abs_error = fabsf(gpu_val - cpu_val);
float rel_error = abs_error / (fabsf(cpu_val) + 1e-6f);

if (abs_error > max_abs_error) {
max_abs_error = abs_error;
max_error_idx = i;
}

if (rel_error > max_rel_error) {
max_rel_error = rel_error;
}
}

// Print comparison results
std::cout << "Maximum absolute error: " << max_abs_error << std::endl;
std::cout << "Maximum relative error: " << max_rel_error << std::endl;

if (max_error_idx >= 0) {
int b = max_error_idx / (num_heads * seq_len * head_dim);
int h = (max_error_idx / (seq_len * head_dim)) % num_heads;
int s = (max_error_idx / head_dim) % seq_len;
int d = max_error_idx % head_dim;

std::cout << "Max error at [b=" << b << ", h=" << h 
<< ", seq=" << s << ", dim=" << d << "]" << std::endl;
std::cout << "  GPU value: " << __half2float(h_output_gpu[max_error_idx]) << std::endl;
std::cout << "  CPU value: " << __half2float(h_output_cpu[max_error_idx]) << std::endl;
}

// Determine test result
const float ERROR_THRESHOLD = 0.01f;  // 1% relative error is acceptable for reduced precision
if (max_rel_error < ERROR_THRESHOLD) {
std::cout << "✓ TEST PASSED" << std::endl;
} else {
std::cout << "✗ TEST FAILED: Error exceeds threshold" << std::endl;

// Print first few values for debugging
std::cout << "\nFirst few values comparison:" << std::endl;
std::cout << "Index | GPU Value | CPU Value | Rel. Error" << std::endl;
std::cout << "------|-----------|-----------|------------" << std::endl;

for (int i = 0; i < std::min(10, (int)h_output_gpu.size()); i++) {
float gpu_val = __half2float(h_output_gpu[i]);
float cpu_val = __half2float(h_output_cpu[i]);
float rel_error = fabsf(gpu_val - cpu_val) / (fabsf(cpu_val) + 1e-6f);

printf("%5d | %9.6f | %9.6f | %10.6f\n", 
i, gpu_val, cpu_val, rel_error);
}
}

// Cleanup
cudaFree(d_query);
cudaFree(d_key);
cudaFree(d_value);
cudaFree(d_output);
}

// Example usage
int main() {
    test_fp8_flash_attention();

    // Example dimensions for a small model
    int batch_size = 2;
    int num_heads = 12;
    int seq_len = 256;
    int head_dim = 64;
    
    // Allocate device memory for inputs and outputs
    half *d_query, *d_key, *d_value, *d_output;
    size_t size = batch_size * num_heads * seq_len * head_dim * sizeof(half);
    
    cudaMalloc(&d_query, size);
    cudaMalloc(&d_key, size);
    cudaMalloc(&d_value, size);
    cudaMalloc(&d_output, size);
    
    // In a real scenario, you would copy data from host to device here
    // ...
    
    // Call the quantized attention function
    fp8_quantized_attention(d_query, d_key, d_value, d_output, 
                           batch_size, num_heads, seq_len, head_dim);
    
    // Copy results back and cleanup
    // ...
    
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
    
    return 0;
}
