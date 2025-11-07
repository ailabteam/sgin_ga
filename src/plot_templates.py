# src/plot_templates.py (v2.1 - Sửa lỗi relative import)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# SỬA LỖI: Thêm dấu chấm để sử dụng relative import
from .publication_style import COLOR_PALETTE

# ===================================================================
# HÀM HELPER (Không đổi)
# ===================================================================
def _setup_ax(ax=None, figsize=(10, 7)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax

# ===================================================================
# CÁC HÀM TEMPLATE VẼ BIỂU ĐỒ (Không đổi)
# ===================================================================

def plot_line_comparison_styled(
    ax, data: pd.DataFrame, x_col: str, y_cols_map: dict,
    title: str, x_label: str, y_label: str
):
    # (Nội dung hàm không đổi)
    markers = ['o', 's', '^', 'D', 'v', 'P']
    linestyles = ['-', '--', ':', '-.']
    colors = [COLOR_PALETTE.get(c, '#000000') for c in ['red', 'blue', 'green', 'orange', 'purple']]
    for i, (y_col, legend_label) in enumerate(y_cols_map.items()):
        linewidth = 3.0 if i == 0 else 2.0
        alpha = 1.0 if i == 0 else 0.8
        ax.plot(
            data[x_col], data[y_col], label=legend_label,
            color=colors[i % len(colors)], marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)], linewidth=linewidth, alpha=alpha,
        )
    ax.set_title(title); ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.legend(title='Methods')
    return ax

def plot_lollipop_chart(
    ax, data: pd.DataFrame, category_col: str, value_col: str,
    title: str, x_label: str, ylim: tuple = None
):
    """
    Vẽ Lollipop Chart, thay thế cho bar chart khi có chênh lệch lớn.
    """
    df_sorted = data.sort_values(by=value_col, ascending=True)
    
    categories = df_sorted[category_col]
    values = df_sorted[value_col]
    
    colors = [COLOR_PALETTE.get(c, '#000000') for c in ['gray', 'blue', 'green', 'red']]
    
    ax.vlines(x=categories, ymin=0, ymax=values, color='grey', alpha=0.4, linewidth=2)
    ax.scatter(categories, values, c=colors, s=150, alpha=1, zorder=3, edgecolors='black', linewidth=1)

    for i, val in enumerate(values):
        ax.text(i, val, f' {val:.4f}', verticalalignment='center', 
                fontdict={'color': 'black', 'size': 12, 'weight':'bold'})

    ax.set_title(title)
    ax.set_ylabel(x_label)
    # Sửa cảnh báo UserWarning bằng cách set ticks trước
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=15, ha='right')
    
    if ylim:
        ax.set_ylim(ylim)
    else:
        min_val = values.min()
        ax.set_ylim(bottom=min_val * 0.8 if min_val < 0.5 else 0.95)
    
    # SỬA LỖI: Đổi 'b' thành 'visible'
    ax.grid(axis='x', visible=False)
    
    return ax


def plot_tsne_with_contours(
    ax, features, labels, title: str,
    palette: dict = None, n_samples: int = 5000
):
    # (Nội dung hàm không đổi)
    print(f"Running t-SNE with contours for '{title}'...")
    if len(features) > n_samples:
        indices = np.random.choice(len(features), n_samples, replace=False)
        features_sample, labels_sample = features[indices], labels[indices]
    else:
        features_sample, labels_sample = features, labels
    scaler = StandardScaler(); features_scaled = scaler.fit_transform(features_sample)
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42)
    features_2d = tsne.fit_transform(features_scaled)
    df_tsne = pd.DataFrame({'dim1': features_2d[:, 0], 'dim2': features_2d[:, 1], 'label': labels_sample})
    if palette is None: palette = {0: COLOR_PALETTE['blue'], 1: COLOR_PALETTE['red']}
    sns.kdeplot(data=df_tsne, x='dim1', y='dim2', hue='label', palette=palette, alpha=0.3, ax=ax, fill=True, levels=5)
    sns.scatterplot(data=df_tsne, x='dim1', y='dim2', hue='label', palette=palette, ax=ax, s=15, alpha=0.6, legend=False)
    ax.set_title(title, weight='bold'); ax.set_xlabel('t-SNE Dimension 1'); ax.set_ylabel('t-SNE Dimension 2')
    ax.set_xticks([]); ax.set_yticks([])
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['No LoS', 'Has LoS'], title='Ground Truth')
    return ax


def plot_distribution_comparison(
    ax, data: pd.DataFrame, x_col: str, y_col: str,
    title: str, x_label: str, y_label: str,
    palette=None
):
    """
    Vẽ biểu đồ Violin để so sánh sự phân phối của một biến qua các nhóm.
    """
    if palette is None:
        colors = [COLOR_PALETTE.get(c, '#000000') for c in ['blue', 'green', 'orange']]
        palette = sns.color_palette(colors)
    
    # SỬA LỖI SEABORN: Gán x_col cho cả x và hue để tương thích với phiên bản mới
    sns.violinplot(
        x=x_col, y=y_col, data=data,
        hue=x_col, # Thêm dòng này
        palette=palette, ax=ax,
        inner='quart',
        linewidth=1.5,
        cut=0,
        legend=False # Tắt legend của violinplot để tránh lặp lại
    )
    
    sns.stripplot(
        x=x_col, y=y_col, data=data,
        color='black', ax=ax,
        size=2,
        alpha=0.2
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # SỬA LỖI MATPLOTLIB: Đổi 'b' thành 'visible'
    ax.grid(axis='x', visible=False)

    return ax

