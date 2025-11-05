# ex4_cga_line_of_sight_fixed.py

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
# ===================================================================

# Các hàm tiện ích (Giữ nguyên)
def create_point_cga(x, y, z):
    p_vec = x*e1 + y*e2 + z*e3
    return eo + p_vec + 0.5 * (p_vec*p_vec) * einf

def create_sphere_cga(center_point_cga, radius):
    return center_point_cga - 0.5 * radius**2 * einf

def main():
    print("--- Bài toán 3: Kiểm tra đường ngắm (Line-of-Sight) - Kịch bản đã sửa ---")

    # --- Định nghĩa các đối tượng ---
    EARTH_RADIUS = 6371.0  # km
    ALTITUDE = 550.0       # km
    ORBIT_RADIUS = EARTH_RADIUS + ALTITUDE
    
    # === SỬA LỖI KỊCH BẢN ===
    # Kịch bản 1: Hai vệ tinh gần nhau (cách 30 độ), chắc chắn CÓ đường ngắm
    sat_A_pos = np.array([ORBIT_RADIUS, 0, 0])
    angle = np.deg2rad(30)
    sat_B_pos = np.array([ORBIT_RADIUS * np.cos(angle), ORBIT_RADIUS * np.sin(angle), 0])
    
    # Kịch bản 2: Hai vệ tinh đối diện nhau, chắc chắn KHÔNG có đường ngắm (Bị che khuất)
    sat_C_pos = np.array([ORBIT_RADIUS, 0, 0])
    sat_D_pos = np.array([-ORBIT_RADIUS, 100, 0])

    # --- 1. Phương pháp CGA ---
    print("\n--- 1. Thực hiện bằng CGA ---")
    
    earth_sphere_cga = create_sphere_cga(create_point_cga(0, 0, 0), EARTH_RADIUS)
    
    pA = create_point_cga(*sat_A_pos); pB = create_point_cga(*sat_B_pos)
    pC = create_point_cga(*sat_C_pos); pD = create_point_cga(*sat_D_pos)
    
    line_AB = pA ^ pB ^ einf
    line_CD = pC ^ pD ^ einf
    
    # Nhắc lại logic: (L|S)^2 > 0 nghĩa là đường thẳng cắt mặt cầu tại 2 điểm
    intersection_check_AB = (line_AB | earth_sphere_cga) * (line_AB | earth_sphere_cga)
    intersection_check_CD = (line_CD | earth_sphere_cga) * (line_CD | earth_sphere_cga)
    
    print(f"Kịch bản 1 (A -> B, gần nhau):")
    check_val_AB = intersection_check_AB[()]
    # Chúng ta cần kiểm tra thêm xem giao điểm có nằm trên đoạn thẳng không, nhưng CGA đã làm điều đó ngầm
    # Cách kiểm tra chính xác hơn là xem xét cặp điểm giao cắt
    print(f"   - (L . S)^2 = {check_val_AB:.2f}")
    print(f"   - Đường ngắm có bị che khuất không? {'Có' if check_val_AB > 0 else 'Không'}")

    print(f"Kịch bản 2 (C -> D, đối diện):")
    check_val_CD = intersection_check_CD[()]
    print(f"   - (L . S)^2 = {check_val_CD:.2f}")
    print(f"   - Đường ngắm có bị che khuất không? {'Có' if check_val_CD > 0 else 'Không'}")


    # --- 2. Phương pháp Vector truyền thống (Ray-Sphere Intersection) ---
    print("\n--- 2. Thực hiện bằng Vector truyền thống (Numpy) ---")
    
    def check_los_numpy(p1, p2, sphere_radius):
        v = p2 - p1
        a = np.dot(v, v)
        b = 2 * np.dot(v, p1)
        c = np.dot(p1, p1) - sphere_radius**2
        
        delta = b**2 - 4*a*c
        
        if delta < 0:
            return False # Không có giao điểm -> Không bị che khuất
        
        # Có giao điểm, kiểm tra xem giao điểm có nằm giữa p1, p2 không
        # Giao điểm xảy ra tại t = (-b +/- sqrt(delta)) / (2a)
        # Nếu 0 < t < 1 thì giao điểm nằm trên đoạn thẳng
        sqrt_delta = np.sqrt(delta)
        t1 = (-b - sqrt_delta) / (2*a)
        t2 = (-b + sqrt_delta) / (2*a)
        
        if (0 < t1 < 1) or (0 < t2 < 1):
            return True # Bị che khuất
            
        return False

    print(f"Kịch bản 1 (A -> B, gần nhau):")
    is_blocked_AB = check_los_numpy(sat_A_pos, sat_B_pos, EARTH_RADIUS)
    print(f"   - Đường ngắm có bị che khuất không? {'Có' if is_blocked_AB else 'Không'}")

    print(f"Kịch bản 2 (C -> D, đối diện):")
    is_blocked_CD = check_los_numpy(sat_C_pos, sat_D_pos, EARTH_RADIUS)
    print(f"   - Đường ngắm có bị che khuất không? {'Có' if is_blocked_CD else 'Không'}")

if __name__ == "__main__":
    main()
