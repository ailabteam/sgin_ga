# src/plot_templates.py
# Chứa các hàm template để vẽ các loại biểu đồ khoa học chất lượng cao.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

# Import bảng màu từ style file của chúng ta
from publication_style import COLOR_PALETTE

# ===================================================================
# HÀM HELPER
# ===================================================================
def _setup_ax(ax=None, figsize=(10, 7)):
    """
    Hàm helper để quản lý việc tạo figure và axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax

# ===================================================================
# CÁC HÀM TEMPLATE VẼ BIỂU ĐỒ
# ===================================================================

def plot_line_comparison(
    ax, data: pd.DataFrame, x_col: str, y_cols_map: dict,
    title: str, x_label: str, y_label: str
):
    """
    Vẽ nhiều đường trên cùng một biểu đồ để so sánh.

    Args:
        ax (matplotlib.axes.Axes): Subplot axis để vẽ lên.
        data (pd.DataFrame): DataFrame chứa dữ liệu.
        x_col (str): Tên cột cho trục X.
        y_cols_map (dict): Dictionary map tên cột Y với tên sẽ hiển thị trên legend.
                           Ví dụ: {'f1_model_a': 'Model A', 'f1_model_b': 'Model B'}
        title (str): Tiêu đề subplot.
        x_label (str): Nhãn trục X.
        y_label (str): Nhãn trục Y.
    """
    markers = ['o', 's', '^', 'D', 'v', 'P']
    linestyles = ['-', '--', ':', '-.']
    colors = list(COLOR_PALETTE.values())

    for i, (y_col, legend_label) in enumerate(y_cols_map.items()):
        ax.plot(
            data[x_col],
            data[y_col],
            label=legend_label,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(title='Methods')
    return ax


def plot_grouped_bar_chart(
    ax, data: pd.DataFrame, category_col: str, value_col: str,
    title: str, y_label: str, ylim: tuple = None
):
    """
    Vẽ biểu đồ cột nhóm.

    Args:
        ax (matplotlib.axes.Axes): Subplot axis để vẽ lên.
        data (pd.DataFrame): DataFrame chứa dữ liệu.
        category_col (str): Tên cột cho các hạng mục trên trục X.
        value_col (str): Tên cột chứa giá trị để vẽ.
        title (str): Tiêu đề subplot.
        y_label (str): Nhãn trục Y.
        ylim (tuple, optional): Giới hạn trục Y, ví dụ (0.9, 1.0).
    """
    categories = data[category_col]
    scores = data[value_col]
    colors = [COLOR_PALETTE[c] for c in ['blue', 'orange', 'green', 'red']]
    
    bars = ax.bar(categories, scores, color=colors, edgecolor='black', width=0.6)

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Feature Sets")
    ax.tick_params(axis='x', rotation=15)
    
    if ylim:
        ax.set_ylim(ylim)
    else:
        # Tự động điều chỉnh giới hạn Y để có không gian cho text
        min_score = scores.min()
        max_score = scores.max()
        ax.set_ylim(bottom=min_score * 0.99, top=max_score * 1.01)

    # Thêm giá trị trên đỉnh mỗi cột
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', 
                va='bottom', ha='center', fontsize=12, weight='bold')

    return ax


def plot_tsne(
    ax, features, labels, title: str,
    palette: dict = None, n_samples: int = 5000
):
    """
    Thực hiện t-SNE và vẽ kết quả scatter plot.

    Args:
        ax (matplotlib.axes.Axes): Subplot axis để vẽ lên.
        features (np.array): Mảng (n_samples, n_features).
        labels (array-like): Nhãn của mỗi mẫu.
        title (str): Tiêu đề subplot.
        palette (dict, optional): Dictionary map nhãn với màu sắc.
        n_samples (int): Số lượng mẫu con để chạy (để tăng tốc).
    """
    print(f"Running t-SNE for '{title}'...")
    if len(features) > n_samples:
        indices = np.random.choice(len(features), n_samples, replace=False)
        features_sample = features[indices]
        labels_sample = labels[indices]
    else:
        features_sample, labels_sample = features, labels
    
    # t-SNE rất nhạy cảm với thang đo, cần chuẩn hóa
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_sample)
    
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42, verbose=0)
    features_2d = tsne.fit_transform(features_scaled)
    
    df_tsne = pd.DataFrame({'dim1': features_2d[:, 0], 'dim2': features_2d[:, 1], 'label': labels_sample})
    
    if palette is None:
        palette = {0: COLOR_PALETTE['blue'], 1: COLOR_PALETTE['red']}
    
    sns.scatterplot(
        data=df_tsne, x='dim1', y='dim2', hue='label',
        palette=palette, ax=ax, s=25, alpha=0.7, edgecolor='k', linewidth=0.3
    )

    ax.set_title(title, weight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_xticks([]); ax.set_yticks([]) # t-SNE axes have no physical meaning
    
    # Tùy chỉnh legend
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['No LoS', 'Has LoS'], title='Ground Truth')
    
    return ax
