# ex3_cga_intersection.py (Phiên bản cuối, dựa trên phân tích toán học chính xác)

# ===================================================================
# THIẾT LẬP MÔI TRƯỜNG
# ===================================================================
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import numpy as np
import clifford

# Khởi tạo không gian CGA (G4,1)
layout, blades = clifford.Cl(4, 1)
e1, e2, e3, e4, e5 = blades['e1'], blades['e2'], blades['e3'], blades['e4'], blades['e5']
ep, en = e4, e5
eo = 0.5 * (en - ep)
einf = en + ep
# ===================================================================

# Các hàm tiện ích
def create_point_cga(x, y, z):
    p_vec = x*e1 + y*e2 + z*e3
    return eo + p_vec + 0.5 * (p_vec*p_vec) * einf

def create_sphere_cga(center_point_cga, radius):
    return center_point_cga - 0.5 * radius**2 * einf

def main():
    print("--- Bài toán 2.1: Tìm giao tuyến của hai vùng phủ vệ tinh ---")

    # --- Định nghĩa các đối tượng ---
    sat1_pos = np.array([-600.0, 0.0, 800.0])
    sat1_radius = 1000.0
    sat2_pos = np.array([600.0, 0.0, 800.0])
    sat2_radius = 1000.0
    
    # --- 1. Phương pháp CGA (Dựa trên phân tích đã được xác thực) ---
    print("\n--- 1. Thực hiện bằng CGA (Phương pháp đã hiệu chỉnh) ---")
    
    # Bước 1: Tạo các đối tượng mặt cầu trong CGA
    sphere1 = create_sphere_cga(create_point_cga(*sat1_pos), sat1_radius)
    sphere2 = create_sphere_cga(create_point_cga(*sat2_pos), sat2_radius)
    
    # Bước 2: Tìm đối tượng vòng tròn giao tuyến bằng phép toán outer product
    intersection_circle = sphere1 ^ sphere2
    
    # Bước 3: Tính bình phương của đối tượng giao tuyến.
    # Theo phân tích, kết quả này bằng -(r^2 * d^2)
    temp_sq = -(intersection_circle * intersection_circle)[()]
    
    # Bước 4: Tính khoảng cách bình phương (d^2) giữa hai tâm
    d_sq = np.sum((sat1_pos - sat2_pos)**2)
    
    # Bước 5: Hiệu chỉnh kết quả bằng cách chia cho d^2 để có r^2
    # Cần kiểm tra d_sq > 0 để tránh chia cho 0 (trường hợp hai mặt cầu trùng tâm)
    if d_sq > 1e-9:
        radius_sq_cga = temp_sq / d_sq
    else:
        radius_sq_cga = -1 # Gán giá trị âm để báo hiệu không có giao tuyến

    print(f"Bán kính bình phương của vòng tròn giao tuyến (tính bằng CGA): {radius_sq_cga:.2f}")
    if radius_sq_cga >= 0:
        print(f"Bán kính của vòng tròn giao tuyến: {np.sqrt(radius_sq_cga):.2f}")
    else:
        print("Không có giao tuyến (hai mặt cầu trùng tâm).")


    # --- 2. Phương pháp Vector truyền thống (Dùng để xác minh) ---
    print("\n--- 2. Thực hiện bằng Vector truyền thống (Numpy) ---")
    
    d = np.linalg.norm(sat1_pos - sat2_pos)
    r1, r2 = sat1_radius, sat2_radius
    
    print(f"Khoảng cách giữa hai tâm vệ tinh: {d:.2f}")
    
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
         print("Hai vùng phủ không giao nhau, chứa nhau, hoặc trùng tâm.")
    else:
        print("Hai vùng phủ có giao nhau.")
        part1 = (d + r1 + r2); part2 = (d + r1 - r2)
        part3 = (d - r1 + r2); part4 = (-d + r1 + r2)
        radius_intersection = (1 / (2*d)) * np.sqrt(part1 * part2 * part3 * part4)
        print(f"Bán kính của vòng tròn giao tuyến (tính bằng hình học): {radius_intersection:.2f}")

if __name__ == "__main__":
    main()
