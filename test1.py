import numpy as np
from clifford.g3c import *
from clifford.tools.g3 import *
from pyganja import *

# --- Mục tiêu: Quay vector v = (1, 0, 0) một góc 90 độ quanh trục z = (0, 0, 1) ---
# Kết quả mong đợi: v_rotated = (0, 1, 0)

# Vector ban đầu
v = np.array([1.0, 0.0, 0.0])
print(f"Vector ban đầu v: {v}\n")

# --- 1. Phương pháp Quaternion (sử dụng numpy) ---
print("--- 1. Sử dụng Quaternion ---")
angle = np.pi / 2  # 90 độ
axis = np.array([0.0, 0.0, 1.0])

# Tạo Quaternion: q = cos(a/2) + sin(a/2) * (ix + jy + kz)
# Ở đây ta biểu diễn phần vector là (x, y, z)
q_scalar = np.cos(angle / 2)
q_vector = np.sin(angle / 2) * axis
q = np.hstack([q_scalar, q_vector]) # w, x, y, z

# Quaternion nghịch đảo (conjugate)
q_conj = np.hstack([q[0], -q[1:]])

# Biểu diễn v dưới dạng pure quaternion (0, v_x, v_y, v_z)
v_quat = np.hstack([0, v])

# Phép quay: v' = q * v * q_conj
# Phép nhân quaternion (Hamilton product)
def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

v_rotated_quat_temp = quat_mult(q, v_quat)
v_rotated_quat = quat_mult(v_rotated_quat_temp, q_conj)

# Lấy phần vector của kết quả
v_rotated_q = v_rotated_quat[1:]
print(f"Vector sau khi quay bằng Quaternion: {v_rotated_q}")
print(f"Kết quả có gần với (0, 1, 0) không? {np.allclose(v_rotated_q, [0, 1, 0])}\n")


# --- 2. Phương pháp Geometric Algebra (sử dụng Clifford) ---
print("--- 2. Sử dụng Geometric Algebra (Rotor) ---")
# Trong GA, trục quay thực chất là một mặt phẳng quay (bivector)
# Quay quanh trục z tương đương với quay trong mặt phẳng xy (e1^e2)
plane_of_rotation = e1^e2

# Tạo Rotor R = exp(-plane * angle/2)
# Công thức Euler cho GA: R = cos(angle/2) - plane*sin(angle/2)
R = np.cos(angle / 2) - plane_of_rotation * np.sin(angle / 2)
print(f"Rotor được tạo ra:\n{R}\n")

# Biểu diễn vector v trong không gian GA
v_ga = v[0]*e1 + v[1]*e2 + v[2]*e3

# Phép quay trong GA rất thanh lịch: v' = R * v * ~R
# ~R là nghịch đảo (reverse) của R
v_rotated_ga = R * v_ga * ~R

print(f"Vector sau khi quay bằng GA Rotor:\n{v_rotated_ga}\n")
# Chuyển kết quả về mảng numpy để so sánh
v_rotated_ga_np = np.array([v_rotated_ga[e1], v_rotated_ga[e2], v_rotated_ga[e3]])
print(f"Vector kết quả dưới dạng numpy: {v_rotated_ga_np}")
print(f"Kết quả có gần với (0, 1, 0) không? {np.allclose(v_rotated_ga_np, [0, 1, 0])}")
