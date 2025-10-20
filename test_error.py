import numpy as np
import matplotlib.pyplot as plt
from struct import unpack, pack

def q_rsqrt(number):
    """
    Fast inverse square root from Quake III, using single-precision float32.
    Implements the bit-level hack + one Newton iteration.
    """
    # Ensure single precision
    number = np.float32(number)
    
    # Precomputes
    threehalfs = np.float32(1.5)
    x2 = number * np.float32(0.5)
    y = number
    
    # Evil bit-level hacking
    i = unpack('I', pack('f', y))[0]  # Reinterpret float as uint32
    i = 0x5f3759df - (i >> 1)         # Magic constant and shift
    y = unpack('f', pack('I', i))[0]  # Reinterpret back to float
    
    # One Newton-Raphson iteration
    y = y * (threehalfs - (x2 * y * y))
    
    return y

# Generate 1000 random single-precision positive floats between 1e-6 and 1e6
# (Avoid 0 to prevent div-by-zero; uniform in log space for better distribution across magnitudes)
np.random.seed(42)  # For reproducibility
num_samples = 1000
x_double = np.random.uniform(low=1e-6, high=1e6, size=num_samples)
x_float32 = x_double.astype(np.float32)  # Cast to single precision for testing

# Compute references and approximations
y_ref = 1.0 / np.sqrt(x_double)  # Double precision reference

y_quake = np.array([q_rsqrt(xi) for xi in x_float32])  # Quake method in float32

# Standard method: 1.0f / sqrtf(x) simulated in float32
y_std = np.array([np.float32(1.0) / np.sqrt(np.float32(xi)) for xi in x_float32])

# Relative errors: (approx - ref) / ref
rel_err_quake = (y_quake - y_ref) / y_ref
rel_err_std = (y_std - y_ref) / y_ref

# Compute and print statistics
print("=== Statistics for Quake III Method ===")
print(f"Min relative error: {np.min(rel_err_quake):.6f}")
print(f"Max relative error: {np.max(rel_err_quake):.6f}")
print(f"Max absolute relative error: {np.max(np.abs(rel_err_quake)):.6f}")
print(f"Mean absolute relative error: {np.mean(np.abs(rel_err_quake)):.6f}")

print("\n=== Statistics for Standard Method (1.0f / sqrtf(x)) ===")
print(f"Min relative error: {np.min(rel_err_std):.6f}")
print(f"Max relative error: {np.max(rel_err_std):.6f}")
print(f"Max absolute relative error: {np.max(np.abs(rel_err_std)):.6f}")
print(f"Mean absolute relative error: {np.mean(np.abs(rel_err_std)):.6f}")

