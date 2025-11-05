# ex4_cga_line_of_sight.py

# ===================================================================
# THIẾT LẬP MÔI TRƯỜNG (Giữ nguyên)
# ===================================================================
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import numpy as np
import clifford

layout, blades = clifford.Cl(4, 1)
e1, e2, e3, e4, e5 = blades['e1'], blades['e2'], blades['e3'], blades['e4'], blades['e5']
ep, en = e4, e5
eo = 0.5 * (en - ep)
einf = en + ep
I5 = e1^e2^e3^e4^e5
# ===================================================================

# Các hàm tiện ích (Giữ nguyên)
def create_point_cga(x, y, z):
    p_vec = x*e1 + y*e2 + z*e3
    return eo + p_vec + 0.5 * (p_vec*p_vec) * einf

def create_sphere_cga(center_point_cga, radius):
    return center_point_cga - 0.5 * radius**2 * einf

def main():
    print("--- Bài toán 3: Kiểm tra đường ngắm (Line-of-Sight) giữa hai vệ tinh ---")

    # --- Định nghĩa các đối tượng ---
    EARTH_RADIUS = 6371.0  # km
    
    # Kịch bản 1: Hai vệ tinh có đường ngắm, không bị che khuất
    sat_A_pos = np.array([EARTH_RADIUS + 550, 0, 0])
    sat_B_pos = np.array([0, EARTH_RADIUS + 550, 0])
    
    # Kịch bản 2: Hai vệ tinh bị Trái Đất che khuất
    sat_C_pos = np.array([EARTH_RADIUS + 550, 0, 0])
    sat_D_pos = np.array([- (EARTH_RADIUS + 550), 100, 0]) # Ở phía đối diện Trái Đất

    # --- 1. Phương pháp CGA ---
    print("\n--- 1. Thực hiện bằng CGA ---")
    
    # Tạo mặt cầu Trái Đất
    earth_center_cga = create_point_cga(0, 0, 0)
    earth_sphere_cga = create_sphere_cga(earth_center_cga, EARTH_RADIUS)
    
    # Tạo các điểm vệ tinh
    pA = create_point_cga(*sat_A_pos)
    pB = create_point_cga(*sat_B_pos)
    pC = create_point_cga(*sat_C_pos)
    pD = create_point_cga(*sat_D_pos)
    
    # Tạo các đường thẳng đi qua các cặp điểm
    # L = P1 ^ P2 ^ einf
    line_AB = pA ^ pB ^ einf
    line_CD = pC ^ pD ^ einf
    
    # Tìm giao điểm giữa đường thẳng và mặt cầu
    # Giao điểm là đối ngẫu của phép toán outer product
    # intersection_pair = (line_AB.dual()) ^ earth_sphere_cga
    # Hoặc cách dễ hiểu hơn: dùng inner product
    # Nếu (line . S)^2 > 0 -> có 2 giao điểm (cắt qua)
    # Nếu (line . S)^2 = 0 -> có 1 giao điểm (tiếp tuyến)
    # Nếu (line . S)^2 < 0 -> không có giao điểm
    
    intersection_check_AB = (line_AB | earth_sphere_cga) * (line_AB | earth_sphere_cga)
    intersection_check_CD = (line_CD | earth_sphere_cga) * (line_CD | earth_sphere_cga)
    
    print("Kịch bản 1 (A -> B):")
    # Lấy giá trị scalar
    check_val_AB = intersection_check_AB[()]
    print(f"   - (L . S)^2 = {check_val_AB:.2f}")
    print(f"   - Đường ngắm có bị che khuất không? {'Có' if check_val_AB > 0 else 'Không'}")

    print("Kịch bản 2 (C -> D):")
    check_val_CD = intersection_check_CD[()]
    print(f"   - (L . S)^2 = {check_val_CD:.2f}")
    print(f"   - Đường ngắm có bị che khuất không? {'Có' if check_val_CD > 0 else 'Không'}")


    # --- 2. Phương pháp Vector truyền thống (Ray-Sphere Intersection) ---
    print("\n--- 2. Thực hiện bằng Vector truyền thống (Numpy) ---")
    
    def check_los_numpy(p1, p2, sphere_radius):
        # p1: vị trí vệ tinh 1, p2: vị trí vệ tinh 2
        direction = p2 - p1
        d_norm = np.linalg.norm(direction)
        direction = direction / d_norm
        
        origin_to_p1 = p1 # Tâm Trái Đất ở (0,0,0)
        
        # Giải phương trình bậc hai: a*t^2 + b*t + c = 0
        a = 1.0 # direction dot direction
        b = 2 * np.dot(direction, origin_to_p1)
        c = np.dot(origin_to_p1, origin_to_p1) - sphere_radius**2
        
        delta = b**2 - 4*a*c
        
        if delta < 0:
            # Không có giao điểm thực -> Không bị che khuất
            return False
        else:
            # Có giao điểm, cần kiểm tra xem giao điểm có nằm giữa p1 và p2 không
            t1 = (-b - np.sqrt(delta)) / (2*a)
            t2 = (-b + np.sqrt(delta)) / (2*a)
            # Nếu có một giá trị t nằm trong khoảng (0, d_norm) thì đoạn thẳng cắt mặt cầu
            if (0 < t1 < d_norm) or (0 < t2 < d_norm):
                return True # Bị che khuất
            return False

    print("Kịch bản 1 (A -> B):")
    is_blocked_AB = check_los_numpy(sat_A_pos, sat_B_pos, EARTH_RADIUS)
    print(f"   - Đường ngắm có bị che khuất không? {'Có' if is_blocked_AB else 'Không'}")

    print("Kịch bản 2 (C -> D):")
    is_blocked_CD = check_los_numpy(sat_C_pos, sat_D_pos, EARTH_RADIUS)
    print(f"   - Đường ngắm có bị che khuất không? {'Có' if is_blocked_CD else 'Không'}")

if __name__ == "__main__":
    main()
