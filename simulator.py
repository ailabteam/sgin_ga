# simulator.py (version 1.1 - Sửa lỗi arange -> linspace)

import numpy as np
import time
from skyfield.api import load, EarthSatellite
from datetime import timedelta

# Import các công cụ CGA đã được xác thực từ bài toán trước
import clifford
# Tắt cảnh báo numba
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

# ===================================================================
# SETUP CGA
# ===================================================================
layout, blades = clifford.Cl(4, 1)
e1, e2, e3, e4, e5 = blades['e1'], blades['e2'], blades['e3'], blades['e4'], blades['e5']
ep, en = e4, e5
eo = 0.5 * (en - ep)
einf = en + ep

def create_point_cga(p_vec_np):
    p_vec = p_vec_np[0]*e1 + p_vec_np[1]*e2 + p_vec_np[2]*e3
    return eo + p_vec + 0.5 * (p_vec*p_vec) * einf

def create_sphere_cga(center_point_cga, radius):
    return center_point_cga - 0.5 * radius**2 * einf

# ===================================================================
# HÀM CHÍNH CỦA BỘ MÔ PHỎNG
# ===================================================================
def main():
    print("--- Bắt đầu Mô phỏng Chòm vệ tinh LEO ---")
    
    # --- 1. Tải dữ liệu và thiết lập tham số ---
    ts = load.timescale()
    try:
        satellites = load.tle_file('iridium.tle')
        print(f"Đã tải thành công {len(satellites)} vệ tinh từ 'iridium.tle'.")
    except Exception as e:
        print(f"Lỗi: Không thể tải file 'iridium.tle'. Hãy chắc chắn bạn đã tải nó.")
        print(f"Chi tiết lỗi: {e}")
        return

    # Lấy một tập hợp con nhỏ để chạy thử nghiệm nhanh
    # N_SATELLITES = 10 
    # satellites = satellites[:N_SATELLITES]
    N_SATELLITES = len(satellites)
    print(f"Sử dụng {N_SATELLITES} vệ tinh cho mô phỏng.")

    EARTH_RADIUS = 6371.0  # km
    
    # Tham số mô phỏng
    start_time = ts.now()
    duration_hours = 1.0
    time_step_seconds = 60
    
    end_time = start_time + timedelta(hours=duration_hours)
    
    # SỬA LỖI: Dùng ts.linspace() thay vì ts.arange() đã bị loại bỏ
    num_steps = int((duration_hours * 3600) / time_step_seconds) + 1
    sim_times = ts.linspace(start_time, end_time, num_steps)
    N_STEPS = len(sim_times)
    
    print(f"Thời gian mô phỏng: {duration_hours} giờ")
    print(f"Bước thời gian: {time_step_seconds} giây")
    print(f"Tổng số bước: {N_STEPS}")

    # --- 2. Chuẩn bị các cấu trúc dữ liệu để lưu kết quả ---
    states_data = []
    connectivity_data = np.zeros((N_STEPS, N_SATELLITES, N_SATELLITES), dtype=np.int8)

    # --- 3. Chạy vòng lặp mô phỏng ---
    print("\n--- Bắt đầu vòng lặp mô phỏng ---")
    
    earth_sphere_cga = create_sphere_cga(create_point_cga(np.array([0,0,0])), EARTH_RADIUS)
    
    start_loop_time = time.time()
    for i, t in enumerate(sim_times):
        if (i + 1) % 10 == 0 or i == N_STEPS - 1 or i == 0:
            print(f"Đang xử lý bước {i+1}/{N_STEPS}...")
            
        geocentric = [s.at(t) for s in satellites]
        positions = np.array([g.position.km for g in geocentric])
        velocities = np.array([g.velocity.km_per_s for g in geocentric])
        
        for j in range(N_SATELLITES):
            sat_id = satellites[j].model.satnum
            pos = positions[j]
            vel = velocities[j]
            states_data.append([i, sat_id, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]])
            
        points_cga = [create_point_cga(pos) for pos in positions]
        
        for j in range(N_SATELLITES):
            for k in range(j + 1, N_SATELLITES):
                line_jk = points_cga[j] ^ points_cga[k] ^ einf
                intersection_check = (line_jk | earth_sphere_cga) * (line_jk | earth_sphere_cga)
                if intersection_check[()] <= 0:
                    connectivity_data[i, j, k] = 1
                    connectivity_data[i, k, j] = 1

    end_loop_time = time.time()
    print(f"--- Vòng lặp mô phỏng hoàn tất sau {end_loop_time - start_loop_time:.2f} giây ---")

    # --- 4. Lưu kết quả ra file ---
    print("\n--- Đang lưu kết quả ra file ---")
    
    states_data_np = np.array(states_data)
    
    np.savez_compressed('simulation_data.npz', 
                        states=states_data_np, 
                        connectivity=connectivity_data,
                        times=sim_times.tt)
                        
    print("Đã lưu dữ liệu vào file 'simulation_data.npz'.")
    print("File chứa 3 mảng: 'states', 'connectivity', và 'times'.")
    print(f"Kích thước mảng states: {states_data_np.shape}")
    print(f"Kích thước mảng connectivity: {connectivity_data.shape}")

if __name__ == "__main__":
    main()
