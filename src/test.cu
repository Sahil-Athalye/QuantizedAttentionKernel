#include "FP8_Attention.h"
#include <stdio.h>
#include <string.h>


int main(int argc, char **argv) {
  // Default parameters
  int batch_size = 1;
  int num_heads  = 1;
  int seq_len    = 8;
  int head_dim   = 8;
  float range    = 200.0f;
  unsigned seed  = 0;

  // Define timing variables
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Parse command-line args
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc)
      batch_size = std::atoi(argv[++i]);
    else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc)
      num_heads = std::atoi(argv[++i]);
    else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc)
      seq_len = std::atoi(argv[++i]);
    else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc)
      head_dim = std::atoi(argv[++i]);
    else if (strcmp(argv[i], "--range") == 0 && i + 1 < argc)
      range = std::atof(argv[++i]);
    else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
      seed = std::stoul(argv[++i]);
    else {
      std::cerr << "Usage: " << argv[0]
                << " [--batch N] [--heads H] [--seq S] [--dim D]"
                << " [--range R] [--seed X]\n";
      return 1;
    }
  }

  std::cout << "Params: batch=" << batch_size << ", heads=" << num_heads
            << ", seq=" << seq_len << ", dim=" << head_dim << ", range=[-"
            << range << ",+" << range << "]"
            << ", seed=" << seed << std::endl;

  size_t N =
      static_cast<size_t>(batch_size) * num_heads * seq_len * head_dim;
  size_t bytes      = N * sizeof(half);
  size_t bytes_full = N * sizeof(float);

  std::vector<half> h_query(N), h_key(N), h_value(N);
  std::vector<float> h_r_query(N), h_r_key(N), h_r_value(N);
  std::vector<half> h_out_gpu(N);
  std::vector<float> h_out_cpu(N);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-range, range);

  // Fill Q, K, V with uniform(-range, +range)
  for (size_t i = 0; i < N; ++i) {
    float q_num = dist(rng);
    float k_num = dist(rng);
    float v_num = dist(rng);

    h_query[i] = __float2half(q_num);
    h_key[i]   = __float2half(k_num);
    h_value[i] = __float2half(v_num);

    h_r_query[i] = q_num;
    h_r_key[i]   = k_num;
    h_r_value[i] = v_num;
  }

  half *d_q, *d_k, *d_v, *d_out;
  float *d_r_q, *d_r_k, *d_r_v, *d_r_out;
  cudaMalloc(&d_q, bytes);
  cudaMalloc(&d_k, bytes);
  cudaMalloc(&d_v, bytes);
  cudaMalloc(&d_out, bytes);

  cudaEventRecord(start);
  cudaMemcpy(d_q, h_query.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_key.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_value.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, bytes);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Time to copy data to GPU: " << milliseconds << " ms" << std::endl;
  //   // allocate refernce gpu memory
  //   cudaMalloc(&d_r_q, bytes_full);
  //   cudaMalloc(&d_r_k, bytes_full);
  //   cudaMalloc(&d_r_v, bytes_full);
  //   cudaMalloc(&d_r_out, bytes_full);

  //   cudaMemcpy(d_r_q, h_r_query.data(), bytes_full,
  //   cudaMemcpyHostToDevice); cudaMemcpy(d_r_k, h_r_key.data(),
  //   bytes_full, cudaMemcpyHostToDevice); cudaMemcpy(d_r_v,
  //   h_r_value.data(), bytes_full, cudaMemcpyHostToDevice);
  //   cudaMemset(d_r_out, 0, bytes_full);

  
  // Run GPU kernel
  cudaEventRecord(start);
  fp8_quantized_attention(d_q, d_k, d_v, d_out, batch_size, num_heads,
                          seq_len, head_dim);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "FP8 kernel execution time: " << milliseconds << " ms" << std::endl;

  cudaEventRecord(start);
  cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Time to copy results back to CPU: " << milliseconds << " ms" << std::endl;

  // Run CPU reference
  reference_attention(h_r_query.data(), h_r_key.data(), h_r_value.data(),
                      h_out_cpu.data(), batch_size, num_heads, seq_len,
                      head_dim);

  // Compute accuracy metrics: CosineSim, Relative L1, RMSE
  double dot = 0.0, norm_o = 0.0, norm_op = 0.0;
  double sum_abs_diff = 0.0, sum_abs_ref = 0.0;
  double sum_sq_err = 0.0;
  for (size_t i = 0; i < N; ++i) {
    float o  = h_out_cpu[i];
    float op = __half2float(h_out_gpu[i]);
    dot += double(o) * double(op);
    norm_o += double(o) * double(o);
    norm_op += double(op) * double(op);
    sum_abs_diff += std::abs(double(o - op));
    sum_abs_ref += std::abs(double(o));
    sum_sq_err += (double(o - op) * double(o - op));
  }
  double cosine_sim = dot / (std::sqrt(norm_o * norm_op) + 1e-12);
  double rel_l1     = sum_abs_diff / (sum_abs_ref + 1e-12);
  double rmse       = std::sqrt(sum_sq_err / double(N));

  std::cout << "Accuracy metrics:\n"
            << "  CosineSim   (↑) = " << cosine_sim << "\n"
            << "  Relative L1 (↓) = " << rel_l1 << "\n"
            << "  RMSE        (↓) = " << rmse << "\n";

  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_out);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
