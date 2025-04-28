mkdir -p debug
# Compile the attention kernel
nvcc -o ./debug/FP8_Attention_Test ./src/FP8_Attention.cu ./src/quantize.cu ./src/test.cu