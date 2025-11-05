# ex1_benchmark_run.py

import numpy as np
import time
from clifford.g3 import e1, e2, e3, layout

# ===================================================================
# CÁC HÀM TÍNH TOÁN (ĐÃ ĐƯỢC KIỂM TRA VÀ HOẠT ĐỘNG)
# ===================================================================

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def rotate_with_quaternion(vector, angle, axis):
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0: return vector
    axis = axis / axis_norm
    
    q_scalar = np.cos(angle / 2); q_vector = np.sin(angle / 2) * axis
    q = np.hstack([q_scalar, q_vector])
    q_conj = np.hstack([q[0], -q[1:]])
    v_quat = np.hstack([0, vector])
    v_rotated_quat = quat_mult(quat_mult(q, v_quat), q_conj)
    return v_rotated_quat[1:]

def rotate_with_ga_stable(vector_ga, R):
    """Hàm GA tối ưu: nhận vector và Rotor đã được tính toán trước."""
    v_rotated_ga = R * vector_ga * ~R
    return v_rotated_ga

# ===================================================================
# HÀM CHÍNH ĐỂ CHẠY BENCHMARK
# ===================================================================

def main():
    """
    Hàm chính để chạy benchmark so sánh hiệu năng giữa Quaternion và GA.
    """
    N = 10000  # Số lượng phép quay cần thực hiện
    print(f"--- Bắt đầu Benchmark: Thực hiện {N} phép quay ngẫu nhiên ---")

    # --- Chuẩn bị dữ liệu ---
    print("Đang tạo dữ liệu ngẫu nhiên...")
    # Tạo N vector ngẫu nhiên
    vectors = np.random.rand(N, 3) * 2 - 1
    
    # Tạo N góc quay ngẫu nhiên từ 0 đến 2*pi
    angles = np.random.rand(N) * 2 * np.pi
    
    # Tạo N trục quay ngẫu nhiên
    axes = np.random.rand(N, 3) * 2 - 1
    
    # --- Benchmark cho Quaternion ---
    print("\n--- 1. Bắt đầu benchmark cho Quaternion ---")
    start_time_q = time.time()
    
    results_q = []
    for i in range(N):
        res = rotate_with_quaternion(vectors[i], angles[i], axes[i])
        results_q.append(res)
        
    end_time_q = time.time()
    duration_q = end_time_q - start_time_q
    print(f"Hoàn thành trong: {duration_q:.6f} giây")
    print(f"Tốc độ: {N / duration_q:.2f} phép quay/giây")

    # --- Benchmark cho Geometric Algebra ---
    print("\n--- 2. Bắt đầu benchmark cho Geometric Algebra ---")
    
    # Tiền xử lý: Chuyển đổi tất cả dữ liệu sang định dạng GA trước
    # Điều này mô phỏng một hệ thống thực tế nơi dữ liệu đã ở sẵn trong không gian GA
    I3 = e1^e2^e3
    vectors_ga = [v[0]*e1 + v[1]*e2 + v[2]*e3 for v in vectors]
    
    rotors = []
    for i in range(N):
        axis_ga = axes[i, 0]*e1 + axes[i, 1]*e2 + axes[i, 2]*e3
        axis_norm = np.linalg.norm(axes[i])
        if axis_norm == 0:
            # Nếu trục là zero, tạo rotor đơn vị (không quay)
            rotors.append(layout.MultiVector(1.0)) 
            continue
            
        plane_of_rotation = -(axis_ga / axis_norm) * I3
        R = np.cos(angles[i] / 2) + plane_of_rotation * np.sin(angles[i] / 2)
        rotors.append(R)

    start_time_ga = time.time()
    
    results_ga = []
    for i in range(N):
        res = rotate_with_ga_stable(vectors_ga[i], rotors[i])
        results_ga.append(res)
        
    end_time_ga = time.time()
    duration_ga = end_time_ga - start_time_ga
    print(f"Hoàn thành trong: {duration_ga:.6f} giây")
    print(f"Tốc độ: {N / duration_ga:.2f} phép quay/giây")
    
    # --- Tổng kết ---
    print("\n--- Kết quả Benchmark ---")
    print(f"Quaternion: {duration_q:.6f} giây")
    print(f"Geometric Algebra: {duration_ga:.6f} giây")
    
    if duration_q < duration_ga:
        print(f"Quaternion nhanh hơn {duration_ga / duration_q:.2f} lần.")
    else:
        print(f"Geometric Algebra nhanh hơn {duration_q / duration_ga:.2f} lần.")

if __name__ == "__main__":
    main()
