# ex2_cga_coverage.py
# Phiên bản đã sửa lỗi logic kiểm tra điểm trong mặt cầu.

# ===================================================================
# THIẾT LẬP MÔI TRƯỜDNG - TẮT CẢNH BÁO NUMBA
# ===================================================================
import warnings
from numba.core.errors import NumbaDeprecationWarning

# Tắt cảnh báo NumbaDeprecationWarning vì chúng ta biết nó vô hại
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# ===================================================================

import numpy as np
import clifford

# ===================================================================
# KHỞI TẠO KHÔNG GIAN CGA (G4,1) VÀ CÁC VECTOR CƠ SỞ
# ===================================================================
# Ký hiệu p=4, q=1, r=0 -> G4,1
layout, blades = clifford.Cl(4, 1)

# Gán các vector cơ sở. Clifford sử dụng e1, e2, ... theo thứ tự
e1, e2, e3, e4, e5 = blades['e1'], blades['e2'], blades['e3'], blades['e4'], blades['e5']

# Định nghĩa các vector cơ sở đặc biệt của CGA theo chuẩn
ep = e4
en = e5
eo = 0.5 * (en - ep)  # Vector cơ sở cho gốc tọa độ
einf = en + ep         # Vector cơ sở cho điểm ở vô cùng
# ===================================================================

def create_point_cga(x, y, z):
    """
    Tạo một điểm trong không gian CGA từ tọa độ 3D.
    Công thức: P = eo + p_vec + 0.5 * |p_vec|^2 * einf
    """
    p_vec = x*e1 + y*e2 + z*e3
    return eo + p_vec + 0.5 * (p_vec*p_vec) * einf

def create_sphere_cga(center_point_cga, radius):
    """
    Tạo một mặt cầu trong không gian CGA.
    Công thức: S = P_center - 0.5 * r^2 * einf
    """
    return center_point_cga - 0.5 * radius**2 * einf

def main():
    """
    Hàm chính thực hiện so sánh giữa CGA và Vector truyền thống.
    """
    print("--- Bài toán 2: Kiểm tra vùng phủ bằng CGA (Đã sửa lỗi logic) ---")

    # --- 1. Phương pháp CGA ---
    print("\n--- 1. Thực hiện bằng CGA ---")
    
    # Định nghĩa các đối tượng hình học
    satellite_pos_3d = np.array([0, 0, 800])
    coverage_radius = 1000
    
    # Người dùng 1: Nằm trong vùng phủ
    user1_pos_3d = np.array([300, 400, 0])
    
    # Người dùng 2: Nằm ngoài vùng phủ
    user2_pos_3d = np.array([900, 800, 0])
    
    # Biểu diễn các đối tượng trong không gian CGA
    satellite_center_cga = create_point_cga(*satellite_pos_3d)
    coverage_sphere_cga = create_sphere_cga(satellite_center_cga, coverage_radius)
    
    user1_cga = create_point_cga(*user1_pos_3d)
    user2_cga = create_point_cga(*user2_pos_3d)
    
    # Phép toán kiểm tra: P . S (inner product)
    # Lấy giá trị scalar của kết quả multivector
    check1 = (user1_cga | coverage_sphere_cga)[()]
    check2 = (user2_cga | coverage_sphere_cga)[()]
    
    # Phân tích kết quả CGA
    print(f"Kiểm tra người dùng 1 ({user1_pos_3d}):")
    print(f"   - Kết quả P . S = {check1:.2f}")
    # SỬA LỖI LOGIC: Quy ước của công thức này là > 0 cho điểm bên trong
    print(f"   - Nằm trong vùng phủ? {'Có' if check1 > 0 else 'Không'}")
    
    print(f"Kiểm tra người dùng 2 ({user2_pos_3d}):")
    print(f"   - Kết quả P . S = {check2:.2f}")
    # SỬA LỖI LOGIC: Quy ước của công thức này là > 0 cho điểm bên trong
    print(f"   - Nằm trong vùng phủ? {'Có' if check2 > 0 else 'Không'}")

    # --- 2. Phương pháp Vector truyền thống ---
    print("\n--- 2. Thực hiện bằng Vector truyền thống (Numpy) ---")
    
    # Tính khoảng cách Euclidean bình phương
    dist_sq_1 = np.sum((user1_pos_3d - satellite_pos_3d)**2)
    dist_sq_2 = np.sum((user2_pos_3d - satellite_pos_3d)**2)
    radius_sq = coverage_radius**2
    
    # Phân tích kết quả Vector
    print(f"Kiểm tra người dùng 1 ({user1_pos_3d}):")
    print(f"   - Khoảng cách bình phương: {dist_sq_1:.2f}")
    print(f"   - Bán kính bình phương: {radius_sq:.2f}")
    print(f"   - Nằm trong vùng phủ? {'Có' if dist_sq_1 < radius_sq else 'Không'}")
    
    print(f"Kiểm tra người dùng 2 ({user2_pos_3d}):")
    print(f"   - Khoảng cách bình phương: {dist_sq_2:.2f}")
    print(f"   - Bán kính bình phương: {radius_sq:.2f}")
    print(f"   - Nằm trong vùng phủ? {'Có' if dist_sq_2 < radius_sq else 'Không'}")

if __name__ == "__main__":
    main()
