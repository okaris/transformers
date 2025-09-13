import torch
import numpy as np

# Let's create a simple tensor that would represent a small attention weight matrix
# Shape: (head_dim * 2, seq_len) = (8, 4) = 32 values total (one block)
values = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [-1.0, -2.0, -3.0, -4.0],
    [0.5, 1.5, 2.5, 3.5],
    [2.0, 3.0, 4.0, 5.0],
    [-2.0, -3.0, -4.0, -5.0],
    [1.5, 2.5, 3.5, 4.5],
    [0.1, 0.2, 0.3, 0.4],
    [-0.1, -0.2, -0.3, -0.4],
], dtype=torch.float32)

# Target shape after reshape: (2 heads, 4 dim, 4 seq_len)
target_shape = (2, 4, 4)

def naive_quantize(x):
    x_flat = x.reshape(-1)
    scale = torch.max(torch.abs(x_flat)) / 7
    quant = torch.round(x_flat / scale).clamp(-8, 7)
    n_bytes = 2 + (x_flat.numel() + 1) // 2
    packed = torch.zeros(n_bytes, dtype=torch.uint8)
    scale_bytes = torch.tensor([scale], dtype=torch.float16).view(torch.uint8)
    packed[0:2] = scale_bytes
    for i in range(0, x_flat.numel()-1, 2):
        byte = ((quant[i].int() & 0xF) << 4) | (quant[i+1].int() & 0xF)
        packed[2 + i//2] = byte.item()
    return packed

def naive_dequantize(packed):
    scale = packed[0:2].view(torch.float16).item()
    values = torch.zeros((packed.numel()-2) * 2, dtype=torch.float32)
    for i in range(packed.numel()-2):
        byte = packed[i+2]
        values[i*2] = ((byte >> 4) & 0xF)
        values[i*2 + 1] = (byte & 0xF)
    values[values > 7] -= 16
    return values * scale

print("Original values:")
print(values)

# First path: quantize -> dequantize -> reshape
print("\nPath 1: quantize -> dequantize -> reshape")
quantized1 = naive_quantize(values)
dequantized1 = naive_dequantize(quantized1)
reshaped1 = dequantized1.reshape(target_shape)
print(reshaped1)

# Second path: quantize -> reshape -> dequantize
print("\nPath 2: quantize -> reshape -> dequantize")
quantized2 = naive_quantize(values)
reshaped_quantized = quantized2.reshape(-1)  # reshape while quantized
dequantized2 = naive_dequantize(reshaped_quantized)
reshaped2 = dequantized2.reshape(target_shape)
print(reshaped2)

# Compare the two paths
print("\nMax difference between paths:", torch.max(torch.abs(reshaped1 - reshaped2)))