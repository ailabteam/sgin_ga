# paper_visualizer.py
# Mục đích: Tạo các bảng biểu và hình ảnh mô tả bộ dữ liệu.
# Output: In ra Table 2 và lưu Figure 2.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def analyze_dataset(group_name):
    """
    Phân tích một file dữ liệu mô phỏng và trả về các thống kê cùng lịch sử kết nối.
    """
    filepath = f'sim_data_{group_name}.npz'
    try:
        data = np.load(filepath, allow_pickle=True)
    except FileNotFoundError:
        print(f"Cảnh báo: Không tìm thấy file '{filepath}'. Bỏ qua.")
        return None, None
        
    connectivity = data['connectivity']
    num_steps, num_sats, _ = connectivity.shape
    
    # Tính số lượng link tại mỗi bước
    num_links_per_step = np.sum(connectivity, axis=(1, 2)) / 2
    max_possible_links = num_sats * (num_sats - 1) / 2
    
    # Tính tỷ lệ kết nối trung bình
    avg_connectivity_ratio = np.mean(num_links_per_step) / max_possible_links if max_possible_links > 0 else 0
    
    stats = {
        'Constellation': group_name.capitalize(),
        'Num Satellites': num_sats,
        'Num Timesteps': num_steps,
        'Total Pairs': int(max_possible_links),
        'Avg Connectivity (%)': f"{avg_connectivity_ratio * 100:.2f}"
    }
    return stats, num_links_per_step

def main():
    """
    Hàm chính để tạo Table 2 và Figure 2 cho bài báo.
    """
    # Đảm bảo thư mục lưu hình ảnh tồn tại
    output_dir = 'paper_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    constellations = ['iridium', 'starlink', 'oneweb']
    
    # --- Tạo Table 2 ---
    print("--- Generating Table 2: Dataset Statistics ---")
    table2_data = []
    all_links_history = {}
    
    for group in constellations:
        stats, links_history = analyze_dataset(group)
        if stats:
            table2_data.append(stats)
            all_links_history[stats['Constellation']] = links_history
    
    if not table2_data:
        print("Không có dữ liệu để xử lý. Hãy chạy simulator_v2.py trước.")
        return

    df_table2 = pd.DataFrame(table2_data)
    print("--- Table 2 ---")
    print(df_table2.to_string(index=False))

    # --- Tạo Figure 2 ---
    print("\n--- Generating Figure 2: Connectivity Over Time ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for name, history in all_links_history.items():
        # Lấy số vệ tinh từ bảng đã tạo
        num_sats_str = df_table2[df_table2['Constellation'] == name]['Num Satellites'].iloc[0]
        label = f'{name} ({num_sats_str} sats)'
        ax.plot(history, marker='.', linestyle='-', markersize=5, label=label)

    ax.set_title('Dynamic Connectivity of LEO Constellations', fontsize=16)
    ax.set_xlabel('Simulation Timestep (60s interval)', fontsize=12)
    ax.set_ylabel('Number of Active Inter-Satellite Links (ISLs)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    
    fig2_filename = os.path.join(output_dir, 'fig_connectivity_over_time.png')
    plt.savefig(fig2_filename, dpi=600) # Lưu với DPI 600
    print(f"Saved Figure 2 to '{fig2_filename}'")

if __name__ == "__main__":
    main()
