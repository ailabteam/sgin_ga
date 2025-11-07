# draw_benchmark_chart.py

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Thêm src vào path và import công cụ
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.publication_style import set_publication_style, COLOR_PALETTE

def main():
    set_publication_style(font_family='serif')
    output_dir = 'paper_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Dữ liệu từ Table 1
    data = {
        'Task': [
            '3D Rotation', '3D Rotation',
            'Sphere-Sphere Int.', 'Sphere-Sphere Int.',
            'LoS Check', 'LoS Check'
        ],
        'Method': [
            'Numpy', 'Clifford',
            'Numpy', 'Clifford',
            'Numpy', 'Clifford'
        ],
        'Time': [
            13.33, 39.33,
            5.41, 9.70,
            6.41, 67.92
        ]
    }
    df = pd.DataFrame(data)
    
    # Pivot dữ liệu để vẽ grouped bar chart
    df_pivot = df.pivot(index='Task', columns='Method', values='Time')
    # Sắp xếp lại cột để Numpy luôn đứng trước
    df_pivot = df_pivot[['Numpy', 'Clifford']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_pivot.plot(
        kind='bar', 
        ax=ax, 
        color=[COLOR_PALETTE['blue'], COLOR_PALETTE['orange']],
        edgecolor='black',
        width=0.6
    )
    
    ax.set_title('Computational Cost of Geometric Operations', weight='bold')
    ax.set_ylabel('Average Execution Time ($\mu$s)')
    ax.set_xlabel('') # Không cần nhãn trục X
    plt.xticks(rotation=0)
    
    # Sử dụng thang đo log để thể hiện rõ sự chênh lệch
    ax.set_yscale('log')
    ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5)

    # Thêm text giá trị trên các cột
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)

    fig_path = os.path.join(output_dir, 'figure_1_microbenchmark.pdf')
    fig.savefig(fig_path)
    print(f"Saved benchmark figure to '{fig_path}'")
    plt.close(fig)

if __name__ == '__main__':
    main()
