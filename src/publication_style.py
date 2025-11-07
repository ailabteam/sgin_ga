# src/publication_style.py

import matplotlib.pyplot as plt
import seaborn as sns

# ===================================================================
# BẢNG MÀU CHÍNH
# Sử dụng bảng màu "Tableau 10", rất rõ ràng và chuyên nghiệp.
# ===================================================================
COLOR_PALETTE = {
    'blue':   '#1f77b4',
    'orange': '#ff7f0e',
    'green':  '#2ca02c',
    'red':    '#d62728',
    'purple': '#9467bd',
    'brown':  '#8c564b',
    'pink':   '#e377c2',
    'gray':   '#7f7f7f',
    'olive':  '#bcbd22',
    'cyan':   '#17becf'
}

# ===================================================================
# HÀM THIẾT LẬP STYLE CHÍNH
# ===================================================================
def set_publication_style(font_family='serif'):
    """
    Thiết lập các thông số rcParams của Matplotlib để tạo ra các figure
    có chất lượng cao, sẵn sàng cho việc công bố khoa học.

    Args:
        font_family (str): 'serif' (vd: Times New Roman) hoặc 
                           'sans-serif' (vd: Arial).
    """
    
    # Sử dụng một style cơ bản của Seaborn làm nền tảng
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- 1. Font Settings ---
    if font_family == 'serif':
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    elif font_family == 'sans-serif':
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['mathtext.fontset'] = 'dejavusans'
    
    # --- 2. Font Size Hierarchy ---
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 20
    plt.rcParams['figure.titleweight'] = 'bold'
    
    # --- 3. Line and Marker Settings ---
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['lines.markeredgewidth'] = 1.0
    
    # --- 4. Axes and Ticks Settings ---
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    
    # --- 5. Legend Settings ---
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.edgecolor'] = 'darkgray'
    plt.rcParams['legend.fancybox'] = False
    
    # --- 6. Grid Settings ---
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'lightgray'
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.linewidth'] = 0.8
    
    # --- 7. Figure Saving Settings ---
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['savefig.format'] = 'pdf' # PDF là định dạng vector tốt nhất
    plt.rcParams['savefig.bbox'] = 'tight'
    
    print("Publication style activated.")
