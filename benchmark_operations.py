# benchmark_operations.py
# Mục đích: Thực hiện micro-benchmark cho các phép toán hình học cơ bản
# để tạo ra Table 1 cho bài báo.

import numpy as np
import timeit
import pandas as pd
import clifford
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

# ===================================================================
# SETUP GA (G3) và CGA (G4,1)
# ===================================================================
layout_g3, blades_g3 = clifford.Cl(3)
g3_e1, g3_e2, g3_e3 = blades_g3['e1'], blades_g3['e2'], blades_g3['e3']

layout_cga, blades_cga = clifford.Cl(4, 1)
cga_e1, cga_e2, cga_e3, cga_ep, cga_en = blades_cga['e1'], blades_cga['e2'], blades_cga['e3'], blades_cga['e4'], blades_cga['e5']
cga_eo = 0.5 * (cga_en - cga_ep)
cga_einf = cga_en + cga_ep

# ===================================================================
# 1. CÁC HÀM CHO TÁC VỤ "PHÉP QUAY 3D"
# ===================================================================

# --- Quaternion (Numpy) ---
def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def rotate_with_quaternion(vector, q, q_conj):
    v_quat = np.hstack([0, vector])
    v_rot_temp = quat_mult(q, v_quat)
    v_rot = quat_mult(v_rot_temp, q_conj)
    return v_rot[1:]

# --- GA Rotor (Clifford) ---
def rotate_with_ga(vector_ga, R):
    return R * vector_ga * ~R

# ===================================================================
# 2. CÁC HÀM CHO TÁC VỤ "GIAO CỦA 2 MẶT CẦU"
# ===================================================================

# --- Vector (Numpy) ---
def sphere_sphere_intersection_numpy(p1, r1, p2, r2):
    d = np.linalg.norm(p1 - p2)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return -1 # Không có giao tuyến hoặc trùng tâm
    part1 = (d + r1 + r2); part2 = (d + r1 - r2)
    part3 = (d - r1 + r2); part4 = (-d + r1 + r2)
    radius_intersection = (1 / (2*d)) * np.sqrt(part1 * part2 * part3 * part4)
    return radius_intersection

# --- CGA (Clifford) ---
def create_point_cga(p_vec_np):
    p_vec = p_vec_np[0]*cga_e1 + p_vec_np[1]*cga_e2 + p_vec_np[2]*cga_e3
    return cga_eo + p_vec + 0.5 * (p_vec*p_vec) * cga_einf

def create_sphere_cga(center_point_cga, radius):
    return center_point_cga - 0.5 * radius**2 * cga_einf

def sphere_sphere_intersection_cga(s1, s2, d_sq):
    intersection_circle = s1 ^ s2
    temp_sq = -(intersection_circle * intersection_circle)[()]
    if d_sq > 1e-9:
        radius_sq_cga = temp_sq / d_sq
        return np.sqrt(radius_sq_cga)
    return -1

# ===================================================================
# 3. CÁC HÀM CHO TÁC VỤ "GIAO ĐOẠN THẲNG - MẶT CẦU"
# ===================================================================

# --- Vector (Numpy) ---
def line_seg_sphere_intersection_numpy(p1, p2, radius):
    v = p2 - p1
    a = np.dot(v, v)
    b = 2 * np.dot(v, p1)
    c = np.dot(p1, p1) - radius**2
    delta = b**2 - 4*a*c
    if delta < 0: return False
    sqrt_delta = np.sqrt(delta)
    t1 = (-b - sqrt_delta) / (2*a)
    t2 = (-b + sqrt_delta) / (2*a)
    if (0 < t1 < 1) or (0 < t2 < 1):
        return True
    return False

# --- CGA (Clifford) ---
def line_seg_sphere_intersection_cga(line, sphere):
    intersection_check = (line | sphere) * (line | sphere)
    return intersection_check[()] > 0

# ===================================================================
# HÀM CHÍNH ĐỂ CHẠY BENCHMARK
# ===================================================================
def main():
    print("--- Generating Table 1: Micro-benchmark of Geometric Operations ---")
    
    num_runs = 10000  # Số lần lặp để lấy trung bình
    
    # --- Setup dữ liệu cho benchmark ---
    # Phép quay
    v_np = np.random.rand(3); angle = np.pi/4; axis = np.random.rand(3); axis /= np.linalg.norm(axis)
    q = np.hstack([np.cos(angle/2), np.sin(angle/2)*axis]); q_conj = np.hstack([q[0], -q[1:]])
    v_ga = v_np[0]*g3_e1 + v_np[1]*g3_e2 + v_np[2]*g3_e3
    R = (np.cos(angle/2) + (g3_e1^g3_e2)*np.sin(angle/2))

    # Giao mặt cầu
    p1_np = np.array([-600.0, 0, 0]); r1 = 1000.0
    p2_np = np.array([600.0, 0, 0]); r2 = 1000.0
    s1_cga = create_sphere_cga(create_point_cga(p1_np), r1)
    s2_cga = create_sphere_cga(create_point_cga(p2_np), r2)
    d_sq = np.sum((p1_np - p2_np)**2)

    # Giao đoạn thẳng-mặt cầu
    sat_C_pos = np.array([6371.0 + 550, 0, 0])
    sat_D_pos = np.array([-(6371.0 + 550), 100, 0])
    earth_rad = 6371.0
    pC_cga = create_point_cga(sat_C_pos); pD_cga = create_point_cga(sat_D_pos)
    line_CD_cga = pC_cga ^ pD_cga ^ cga_einf
    earth_sphere_cga = create_sphere_cga(create_point_cga(np.array([0,0,0])), earth_rad)
    
    # --- Chạy benchmark ---
    results = []
    
    # 1. Phép quay
    t_quat = timeit.timeit(lambda: rotate_with_quaternion(v_np, q, q_conj), number=num_runs)
    t_ga_rot = timeit.timeit(lambda: rotate_with_ga(v_ga, R), number=num_runs)
    results.append({'Task': '3D Rotation', 'Method': 'Quaternion (Numpy)', 'Time (µs)': (t_quat / num_runs) * 1e6})
    results.append({'Task': '3D Rotation', 'Method': 'GA Rotor (Clifford)', 'Time (µs)': (t_ga_rot / num_runs) * 1e6})

    # 2. Giao mặt cầu
    t_sph_np = timeit.timeit(lambda: sphere_sphere_intersection_numpy(p1_np, r1, p2_np, r2), number=num_runs)
    t_sph_cga = timeit.timeit(lambda: sphere_sphere_intersection_cga(s1_cga, s2_cga, d_sq), number=num_runs)
    results.append({'Task': 'Sphere-Sphere Intersection', 'Method': 'Vector (Numpy)', 'Time (µs)': (t_sph_np / num_runs) * 1e6})
    results.append({'Task': 'Sphere-Sphere Intersection', 'Method': 'CGA (Clifford)', 'Time (µs)': (t_sph_cga / num_runs) * 1e6})

    # 3. Giao đoạn thẳng-mặt cầu
    t_los_np = timeit.timeit(lambda: line_seg_sphere_intersection_numpy(sat_C_pos, sat_D_pos, earth_rad), number=num_runs)
    t_los_cga = timeit.timeit(lambda: line_seg_sphere_intersection_cga(line_CD_cga, earth_sphere_cga), number=num_runs)
    results.append({'Task': 'LineSeg-Sphere Intersection', 'Method': 'Vector (Numpy)', 'Time (µs)': (t_los_np / num_runs) * 1e6})
    results.append({'Task': 'LineSeg-Sphere Intersection', 'Method': 'CGA (Clifford)', 'Time (µs)': (t_los_cga / num_runs) * 1e6})

    # --- In bảng kết quả ---
    df = pd.DataFrame(results)
    df['Time (µs)'] = df['Time (µs)'].round(2)
    print("\n--- Table 1 ---")
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
