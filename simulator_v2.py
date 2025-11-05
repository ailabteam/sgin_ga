# simulator_v2.py

import numpy as np
import time
from skyfield.api import load
from datetime import timedelta
import argparse  # Thư viện để xử lý tham số dòng lệnh
import clifford
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

# --- SETUP CGA (Không đổi) ---
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

def run_simulation(group_name, tle_file, duration_hours, time_step_seconds, max_sats, output_file):
    print(f"--- Bắt đầu Mô phỏng cho: {group_name} ---")
    
    ts = load.timescale()
    try:
        satellites = load.tle_file(tle_file)
        print(f"Đã tải {len(satellites)} vệ tinh từ '{tle_file}'.")
    except Exception as e:
        print(f"Lỗi: Không thể tải file '{tle_file}'. Lỗi: {e}")
        return

    if max_sats > 0 and len(satellites) > max_sats:
        print(f"Giới hạn số lượng vệ tinh xuống còn {max_sats}.")
        satellites = satellites[:max_sats]
    
    N_SATELLITES = len(satellites)
    EARTH_RADIUS = 6371.0
    
    start_time = ts.now()
    end_time = start_time + timedelta(hours=duration_hours)
    num_steps = int((duration_hours * 3600) / time_step_seconds) + 1
    sim_times = ts.linspace(start_time, end_time, num_steps)
    N_STEPS = len(sim_times)
    
    print(f"Tham số: {duration_hours} giờ, bước {time_step_seconds}s, {N_STEPS} bước.")

    states_data = []
    connectivity_data = np.zeros((N_STEPS, N_SATELLITES, N_SATELLITES), dtype=np.int8)

    print("\n--- Bắt đầu vòng lặp mô phỏng ---")
    earth_sphere_cga = create_sphere_cga(create_point_cga(np.array([0,0,0])), EARTH_RADIUS)
    start_loop_time = time.time()
    
    for i, t in enumerate(sim_times):
        if (i * 10 // N_STEPS) != ((i - 1) * 10 // N_STEPS) or i == N_STEPS - 1:
            print(f"Tiến trình: {((i+1)/N_STEPS)*100:.1f}%")
            
        geocentric = [s.at(t) for s in satellites]
        positions = np.array([g.position.km for g in geocentric])
        velocities = np.array([g.velocity.km_per_s for g in geocentric])
        
        for j in range(N_SATELLITES):
            states_data.append([i, satellites[j].model.satnum, *positions[j], *velocities[j]])
            
        points_cga = [create_point_cga(pos) for pos in positions]
        
        for j in range(N_SATELLITES):
            for k in range(j + 1, N_SATELLITES):
                line_jk = points_cga[j] ^ points_cga[k] ^ einf
                intersection_check = (line_jk | earth_sphere_cga) * (line_jk | earth_sphere_cga)
                if intersection_check[()] <= 0:
                    connectivity_data[i, j, k] = 1
                    connectivity_data[i, k, j] = 1

    end_loop_time = time.time()
    print(f"--- Hoàn tất sau {end_loop_time - start_loop_time:.2f} giây ---")

    print(f"\n--- Đang lưu kết quả vào '{output_file}' ---")
    states_data_np = np.array(states_data)
    np.savez_compressed(output_file, 
                        states=states_data_np, 
                        connectivity=connectivity_data,
                        times=sim_times.tt)
    print("Lưu thành công.")

def main():
    parser = argparse.ArgumentParser(description="Chạy mô phỏng động cho các chòm vệ tinh LEO.")
    parser.add_argument('--group', type=str, required=True, choices=['iridium', 'starlink', 'oneweb'],
                        help="Tên chòm vệ tinh để chạy.")
    parser.add_argument('--duration', type=float, default=1.0, help="Thời gian mô phỏng (giờ).")
    parser.add_argument('--timestep', type=int, default=60, help="Bước thời gian (giây).")
    parser.add_argument('--max_sats', type=int, default=50, help="Giới hạn số vệ tinh để chạy nhanh (0 = tất cả).")
    
    args = parser.parse_args()
    
    tle_files = {
        'iridium': 'iridium.tle',
        'starlink': 'starlink.tle',
        'oneweb': 'oneweb.tle'
    }
    output_file = f"sim_data_{args.group}.npz"
    
    run_simulation(args.group, tle_files[args.group], args.duration, args.timestep, args.max_sats, output_file)

if __name__ == "__main__":
    main()
