#ifndef QUANTIZE_H
#define QUANTIZE_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <iomanip>


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
                                    float scale, int size);

// Helper function to dequantize from FP8_E4M3 to FP16
__device__ void dequantize_fp8_to_fp16(const fp8_e4m3* input, half* output, 
                                      float scale, int size);

#endif