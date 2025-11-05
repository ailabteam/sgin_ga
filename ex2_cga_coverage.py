# ex2_cga_coverage.py

# ===================================================================
# THIẾT LẬP MÔI TRƯỜNG - TẮT CẢNH BÁO NUMBA
# ===================================================================
import warnings
from numba.core.errors import NumbaDeprecationWarning

# Chúng ta biết clifford dùng hàm cũ của numba.
# Việc này là bình thường và vô hại. Hãy tắt cảnh báo này đi.
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# ===================================================================

import numpy as np
from clifford.g4c import layout, e1, e2, e3, ep, en, eo # Nhập các vector cơ sở của CGA

def create_point(x, y, z):
    """Tạo một điểm trong không gian CGA từ tọa độ 3D."""
    # P = x*e1 + y*e2 + z*e3 + 0.5*(x^2+y^2+z^2)*en + ep
    return layout.Vector(x*e1 + y*e2 + z*e3 + 0.5*(x**2+y**2+z**2)*en + ep)

def create_sphere(center_point, radius):
    """Tạo một mặt cầu trong không gian CGA."""
    # S = P - 0.5*r^2*en
    return center_point - 0.5*radius**2*en

def main():
    print("--- Bài toán 2: Kiểm tra vùng phủ bằng Conformal Geometric Algebra (CGA) ---")

    # --- 1. Phương pháp CGA ---
    print("\n--- 1. Thực hiện bằng CGA ---")
    
    # Định nghĩa các đối tượng
    # Vệ tinh ở tọa độ (0, 0, 800) km, vùng phủ có bán kính 1000 km
    satellite_pos_3d = np.array([0, 0, 800])
    coverage_radius = 1000
    
    # Người dùng 1: ở (300, 400, 0), nằm trong vùng phủ
    user1_pos_3d = np.array([300, 400, 0])
    
    # Người dùng 2: ở (900, 800, 0), nằm ngoài vùng phủ
    user2_pos_3d = np.array([900, 800, 0])
    
    # Biểu diễn các đối tượng trong không gian CGA
    # Vùng phủ là một mặt cầu
    satellite_center_cga = create_point(*satellite_pos_3d)
    coverage_sphere_cga = create_sphere(satellite_center_cga, coverage_radius)
    
    # Người dùng là các điểm
    user1_cga = create_point(*user1_pos_3d)
    user2_cga = create_point(*user2_pos_3d)
    
    # Phép toán kiểm tra
    # Một điểm P nằm trong mặt cầu S nếu P . S < 0
    check1 = (user1_cga | coverage_sphere_cga)[()] # Lấy giá trị scalar
    check2 = (user2_cga | coverage_sphere_cga)[()]
    
    print(f"Kiểm tra người dùng 1 ({user1_pos_3d}):")
    print(f"   - Kết quả P . S = {check1:.2f}")
    print(f"   - Nằm trong vùng phủ? {'Có' if check1 < 0 else 'Không'}")
    
    print(f"Kiểm tra người dùng 2 ({user2_pos_3d}):")
    print(f"   - Kết quả P . S = {check2:.2f}")
    print(f"   - Nằm trong vùng phủ? {'Có' if check2 < 0 else 'Không'}")

    # --- 2. Phương pháp Vector truyền thống ---
    print("\n--- 2. Thực hiện bằng Vector truyền thống (Numpy) ---")
    
    # Tính khoảng cách Euclidean bình phương
    dist_sq_1 = np.sum((user1_pos_3d - satellite_pos_3d)**2)
    dist_sq_2 = np.sum((user2_pos_3d - satellite_pos_3d)**2)
    
    radius_sq = coverage_radius**2
    
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
