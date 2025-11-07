# analyze_results.py (v1.1 - Sửa lỗi KeyError)
# Đọc tất cả các file CSV kết quả, tính trung bình và độ lệch chuẩn.

import pandas as pd
import os

def analyze():
    results_dir = 'results_multiple_runs'
    
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' not found. Please run 'run_experiments.sh' first.")
        return

    all_dfs = []
    for filename in os.listdir(results_dir):
        if filename.startswith('results_seed_') and filename.endswith('.csv'):
            filepath = os.path.join(results_dir, filename)
            df = pd.read_csv(filepath)
            seed = int(filename.split('_')[-1].replace('.csv', ''))
            df['seed'] = seed
            all_dfs.append(df)
            
    if not all_dfs:
        print("No result files found in the directory.")
        return
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Tính toán trung bình và độ lệch chuẩn
    summary_df = combined_df.groupby(['Model', 'Features']).agg(
        mean_f1=('F1 Score', 'mean'),
        std_f1=('F1 Score', 'std'),
        mean_auc_roc=('AUC-ROC', 'mean'),
        std_auc_roc=('AUC-ROC', 'std'),
        mean_precision=('Precision', 'mean'),
        std_precision=('Precision', 'std'),
        mean_recall=('Recall', 'mean'),
        std_recall=('Recall', 'std'),
    ).reset_index()
    
    print("--- Final Summary Table (Mean ± Std over multiple runs) ---")
    
    # SỬA LỖI: Đổi tên metric 'auc-roc' thành 'auc_roc' để nhất quán
    metrics_to_format = {
        'f1': 'F1 Score',
        'auc_roc': 'AUC-ROC',
        'precision': 'Precision',
        'recall': 'Recall'
    }

    for key, name in metrics_to_format.items():
        summary_df[f'{name} (Mean ± Std)'] = summary_df.apply(
            lambda row: f"{row[f'mean_{key}']:.4f} ± {row[f'std_{key}']:.4f}",
            axis=1
        )
    
    # SỬA LỖI: Sử dụng đúng tên cột đã tạo
    final_columns = ['Model', 'Features'] + [f'{name} (Mean ± Std)' for name in metrics_to_format.values()]
    final_table = summary_df[final_columns]
    
    print(final_table.to_string(index=False))

if __name__ == '__main__':
    analyze()
