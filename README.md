# FP8-Attention

## Overview

FP8-Attention provides optimized CUDA kernels for transformer attention mechanisms using FP8 quantization. This project aims to accelerate Large Language Model inference while reducing memory requirements through efficient attention computation.

## Key Features

- **FP8 Quantization**: Reduced precision for Query and Key matrices
- **FlashAttention-style Tiling**: Memory-efficient computation avoiding full attention matrix materialization  
- **Online Softmax**: Incremental softmax computation with minimal memory overhead
- **Mixed Precision**: Strategic use of FP16 and FP8 for optimal accuracy-performance balance

## Benefits

- Reduced memory consumption
- Faster inference speeds
- Lower DRAM traffic
- Minimal accuracy loss

## Project Goals

This project was developed to explore the impact of quantization on transformer attention mechanisms, with particular focus on making LLM inference more efficient on consumer hardware. By implementing FlashAttention-style tiling alongside FP8 precision, we address both computational and memory bottlenecks in modern language models.

## Contributors

- Yiqun Du
- Sahil Athalye

## License

MIT License
