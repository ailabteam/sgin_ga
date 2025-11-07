# draw_paper_figures.py
# Script tổng hợp để tạo tất cả các hình ảnh cho bài báo từ dữ liệu đã được xử lý.

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Thêm thư mục src vào Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import các công cụ vẽ từ src
from src.publication_style import set_publication_style, COLOR_PALETTE
from src.plot_templates import (
    plot_distribution_comparison,
    plot_tsne_with_contours,
    plot_lollipop_chart,
    plot_line_comparison_styled,
)

# --- CÁC HÀM VẼ CHO TỪNG FIGURE ---

def draw_figure_2(output_dir, constellations):
    """Vẽ Figure 2: Biểu đồ phân phối kết nối."""
    print("--- Generating Figure 2: Connectivity Distribution (Violin Plot) ---")
    all_links_data = []
    for group in constellations:
        try:
            data = np.load(f'sim_data_{group}.npz', allow_pickle=True)
            connectivity = data['connectivity']
            num_links_per_step = np.sum(connectivity, axis=(1, 2)) / 2
            for num_links in num_links_per_step:
                all_links_data.append({'Constellation': group.capitalize(), 'Active ISLs': num_links})
        except FileNotFoundError:
            print(f"  - Warning: sim_data_{group}.npz not found.")
            continue
    
    df_links = pd.DataFrame(all_links_data)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_distribution_comparison(
        ax=ax,
        data=df_links,
        x_col='Constellation',
        y_col='Active ISLs',
        title='Distribution of Active ISLs Across Constellations',
        x_label='Constellation',
        y_label='Number of Active Inter-Satellite Links (ISLs)'
    )
    
    fig_path = os.path.join(output_dir, 'figure_2_connectivity_distribution.pdf')
    fig.savefig(fig_path)
    print(f"  -> Saved Figure 2 to '{fig_path}'")
    plt.close(fig)

def draw_figure_3(output_dir):
    """Vẽ Figure 3: So sánh không gian đặc trưng t-SNE."""
    print("\n--- Generating Figure 3: t-SNE Feature Space Visualization ---")
    try:
        df_tsne_vec = pd.read_csv('results/figure_3_tsne_vector_data.csv')
        df_tsne_ga = pd.read_csv('results/figure_3_tsne_ga_data.csv')
    except FileNotFoundError:
        print("  - Error: t-SNE data files not found in 'results/'. Please run 'generate_tsne_data.py'.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    # Panel (a)
    plot_tsne_with_contours(ax1, df_tsne_vec[['dim1', 'dim2']].values, df_tsne_vec['label'].values, "(a) Traditional Vector Features")
    
    # Panel (b)
    plot_tsne_with_contours(ax2, df_tsne_ga[['dim1', 'dim2']].values, df_tsne_ga['label'].values, "(b) Geometric Algebra Features")

    fig.suptitle("t-SNE Visualization of Feature Spaces for LoS Classification", weight='bold')
    
    fig_path = os.path.join(output_dir, 'figure_3_tsne_feature_space.pdf')
    fig.savefig(fig_path)
    print(f"  -> Saved Figure 3 to '{fig_path}'")
    plt.close(fig)

def draw_figure_4(output_dir):
    """Vẽ Figure 4: Ablation Study (Lollipop Chart)."""
    print("\n--- Generating Figure 4: Feature Ablation Study (Lollipop Chart) ---")
    try:
        data = pd.read_csv('results/figure_4_ablation_data.csv')
    except FileNotFoundError:
        print("  - Error: Ablation study data not found. Please run 'train_ablation_study.py'.")
        return
        
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_lollipop_chart(
        ax=ax,
        data=data,
        category_col='Feature_Set',
        value_col='F1_Score',
        title='Ablation Study of Feature Components for LSTM Model',
        x_label='F1 Score on Test Set' # Nhãn trục Y thực ra là X label cho lollipop
    )
    
    fig_path = os.path.join(output_dir, 'figure_4_ablation_study.pdf')
    fig.savefig(fig_path)
    print(f"  -> Saved Figure 4 to '{fig_path}'")
    plt.close(fig)

def draw_figure_5(output_dir):
    """Vẽ Figure 5: Phân tích hội tụ."""
    print("\n--- Generating Figure 5: Convergence Analysis ---")
    try:
        data = pd.read_csv('results/figure_5_convergence_data.csv')
    except FileNotFoundError:
        print("  - Error: Convergence history not found. Please run 'train_convergence_analysis.py'.")
        return
        
    data['Method'] = data['Model'] + ' (' + data['Features'] + ')'
    pivot_data = data.pivot_table(index='Epoch', columns='Method', values='Validation_F1').reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_cols_map = {
        'LSTM (Hybrid)': 'LSTM (Hybrid)',
        'MLP (Vector)': 'MLP (Vector)',
        'LSTM (GA)': 'LSTM (GA)',
        'LSTM (Vector)': 'LSTM (Vector)',
    }
    
    # Sắp xếp lại thứ tự các cột để vẽ đường quan trọng nhất lên trên
    plot_order = ['LSTM (Hybrid)', 'MLP (Vector)', 'LSTM (GA)', 'LSTM (Vector)']
    ordered_y_cols = {col: y_cols_map[col] for col in plot_order if col in y_cols_map}

    plot_line_comparison_styled(
        ax=ax,
        data=pivot_data,
        x_col='Epoch',
        y_cols_map=ordered_y_cols,
        title='Convergence Analysis over 30 Epochs',
        x_label='Epoch',
        y_label='Validation F1 Score'
    )
    ax.set_xticks(np.arange(0, 31, 5))
    
    fig_path = os.path.join(output_dir, 'figure_5_convergence.pdf')
    fig.savefig(fig_path)
    print(f"  -> Saved Figure 5 to '{fig_path}'")
    plt.close(fig)
    
def draw_figure_6(output_dir):
    """Vẽ Figure 6: Phân tích độ nhạy."""
    print("\n--- Generating Figure 6: Sensitivity Analysis ---")
    try:
        data = pd.read_csv('results/figure_6_sensitivity_data.csv')
    except FileNotFoundError:
        print("  - Error: Sensitivity analysis data not found. Please run 'train_sensitivity_analysis.py'.")
        return

    data['Method'] = data['Model'] + ' (' + data['Features'] + ')'
    pivot_data = data.pivot_table(index='Horizon (minutes)', columns='Method', values='F1_Score').reset_index()

    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_cols_map = {
        'LSTM (Hybrid)': 'LSTM (Hybrid)',
        'MLP (Vector)': 'MLP (Vector)',
        'LSTM (GA)': 'LSTM (GA)',
        'LSTM (Vector)': 'LSTM (Vector)',
    }
    
    plot_order = ['LSTM (Hybrid)', 'MLP (Vector)', 'LSTM (GA)', 'LSTM (Vector)']
    ordered_y_cols = {col: y_cols_map[col] for col in plot_order if col in y_cols_map}

    plot_line_comparison_styled(
        ax=ax,
        data=pivot_data,
        x_col='Horizon (minutes)',
        y_cols_map=ordered_y_cols,
        title='Model Performance Sensitivity to Prediction Horizon',
        x_label='Prediction Horizon (minutes)',
        y_label='F1 Score on Test Set'
    )
    ax.set_xticks(data['Horizon (minutes)'].unique())
    
    fig_path = os.path.join(output_dir, 'figure_6_sensitivity_analysis.pdf')
    fig.savefig(fig_path)
    print(f"  -> Saved Figure 6 to '{fig_path}'")
    plt.close(fig)

def main():
    # Kích hoạt style chuyên nghiệp, dùng font Serif cho paper
    set_publication_style(font_family='serif')
    
    output_dir = 'paper_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Chạy các hàm vẽ
    draw_figure_2(output_dir, ['iridium', 'starlink', 'oneweb'])
    draw_figure_3(output_dir)
    draw_figure_4(output_dir)
    draw_figure_5(output_dir)
    draw_figure_6(output_dir)
    
    print("\nAll figures have been generated and saved to the 'paper_figures' directory.")

if __name__ == '__main__':
    main()
