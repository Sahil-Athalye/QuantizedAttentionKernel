mkdir -p debug
# Compile the attention kernel
nvcc -o ./debug/FP8_Attention ./src/FP8_Attention.cu