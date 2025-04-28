#include "quantize.h"
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
