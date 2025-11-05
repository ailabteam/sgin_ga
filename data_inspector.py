# data_inspector.py

import numpy as np
import matplotlib.pyplot as plt

def main():
    print("--- Trình kiểm tra Dữ liệu Mô phỏng ---")
    
    # --- 1. Tải dữ liệu ---
    try:
        data = np.load('simulation_data.npz')
        states = data['states']
        connectivity = data['connectivity']
        times = data['times']
        print("Đã tải thành công dữ liệu từ 'simulation_data.npz'.\n")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'simulation_data.npz'.")
        print("Hãy chạy 'simulator.py' trước.")
        return
        
    # --- 2. In thông tin thống kê ---
    num_steps, num_sats, _ = connectivity.shape
    print(f"Thông tin chung:")
    print(f" - Số bước thời gian: {num_steps}")
    print(f" - Số lượng vệ tinh: {num_sats}")
    
    # Tính toán số lượng kết nối (links) tại mỗi bước thời gian
    # Mỗi link (i, j) được đếm 2 lần (i,j) và (j,i), nên ta chia cho 2
    num_links_per_step = np.sum(connectivity, axis=(1, 2)) / 2
    
    print("\nThống kê về kết nối (links):")
    print(f" - Tổng số link tối đa có thể có: {num_sats * (num_sats - 1) / 2}")
    print(f" - Số link trung bình qua các bước: {np.mean(num_links_per_step):.2f}")
    print(f" - Số link ít nhất trong một bước: {np.min(num_links_per_step)}")
    print(f" - Số link nhiều nhất trong một bước: {np.max(num_links_per_step)}")
    
    # In ra trạng thái của vệ tinh đầu tiên tại bước đầu tiên
    first_sat_first_step = states[0, :]
    print("\nDữ liệu trạng thái mẫu (Vệ tinh đầu tiên, bước đầu tiên):")
    print(f" - Step: {int(first_sat_first_step[0])}, SatID: {int(first_sat_first_step[1])}")
    print(f" - Position (x,y,z): ({first_sat_first_step[2]:.2f}, {first_sat_first_step[3]:.2f}, {first_sat_first_step[4]:.2f}) km")
    print(f" - Velocity (vx,vy,vz): ({first_sat_first_step[5]:.2f}, {first_sat_first_step[6]:.2f}, {first_sat_first_step[7]:.2f}) km/s")

    # --- 3. Trực quan hóa dữ liệu ---
    print("\n--- Đang tạo biểu đồ... ---")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(num_links_per_step, marker='o', linestyle='-', markersize=4)
    
    ax.set_title(f'Số lượng Liên kết (ISL) trong Chòm vệ tinh Iridium ({num_sats} vệ tinh) theo Thời gian', fontsize=16)
    ax.set_xlabel('Bước thời gian (mỗi bước = 60 giây)', fontsize=12)
    ax.set_ylabel('Tổng số liên kết có LoS', fontsize=12)
    ax.grid(True)
    
    # Lưu biểu đồ ra file
    figure_filename = 'isl_connectivity_over_time.png'
    plt.savefig(figure_filename, dpi=300) # dpi=300 là đủ nét cho báo cáo
    
    print(f"Đã lưu biểu đồ vào file '{figure_filename}'.")
    print("Bạn có thể tải file này về máy tính cá nhân để xem.")

if __name__ == "__main__":
    main()
