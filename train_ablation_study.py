# train_ablation_study.py
# Mục đích: Tạo Figure 4 - Thực hiện Ablation Study để đánh giá tầm quan trọng
# của từng thành phần trong bộ đặc trưng GA/Hybrid.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import clifford
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# --- KIẾN TRÚC AI (Chỉ cần LSTM) ---
class LSTM_Predictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTM_Predictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x); out = self.fc(out[:, -1, :]); return self.sigmoid(out)

# --- HÀM CHUẨN BỊ DỮ LIỆU (Cập nhật để linh hoạt hơn) ---
def process_single_constellation(states, connectivity, feature_type, sequence_length=5, prediction_horizon=1):
    # (Hàm này giữ nguyên như phiên bản hoạt động trước đó)
    num_steps, num_sats, _ = connectivity.shape
    unique_sat_ids = np.unique(states[:, 1])
    sat_id_to_idx = {int(sat_id): i for i, sat_id in enumerate(unique_sat_ids)}
    if num_sats != len(unique_sat_ids): return [], []
    state_tensor = np.zeros((num_steps, num_sats, 6))
    for row in states:
        step, sat_id = int(row[0]), int(row[1])
        if sat_id in sat_id_to_idx:
            state_tensor[step, sat_id_to_idx[sat_id], :] = row[2:]
    if feature_type in ['bivector_only', 'relative_only']: feature_dim = 6
    else: feature_dim = 12
    feature_tensor = np.zeros((num_steps, num_sats, num_sats, feature_dim))
    layout_g3, blades_g3 = clifford.Cl(3)
    e1, e2, e3 = blades_g3['e1'], blades_g3['e2'], blades_g3['e3']
    for t in range(num_steps):
        for j in range(num_sats):
            for k in range(j + 1, num_sats):
                state_j_np = state_tensor[t, j, :]; state_k_np = state_tensor[t, k, :]
                pos_j_np, vel_j_np = state_j_np[:3], state_j_np[3:]
                pos_k_np, vel_k_np = state_k_np[:3], state_k_np[3:]
                pos_j_ga = pos_j_np[0]*e1 + pos_j_np[1]*e2 + pos_j_np[2]*e3
                vel_j_ga = vel_j_np[0]*e1 + vel_j_np[1]*e2 + vel_j_np[2]*e3
                pos_k_ga = pos_k_np[0]*e1 + pos_k_np[1]*e2 + pos_k_np[2]*e3
                vel_k_ga = vel_k_np[0]*e1 + vel_k_np[1]*e2 + vel_k_np[2]*e3
                bivector_j = pos_j_ga ^ vel_j_ga; bivector_k = pos_k_ga ^ vel_k_ga
                b_j_coeffs = np.array([bivector_j[e1^e2], bivector_j[e1^e3], bivector_j[e2^e3]])
                b_k_coeffs = np.array([bivector_k[e1^e2], bivector_k[e1^e3], bivector_k[e2^e3]])
                if feature_type == 'bivector_only':
                    feature = np.concatenate([b_j_coeffs, b_k_coeffs])
                    feature_kj = np.concatenate([b_k_coeffs, b_j_coeffs])
                elif feature_type == 'relative_only':
                    rel_pos_np = pos_j_np - pos_k_np; rel_vel_np = vel_j_np - vel_k_np
                    feature = np.concatenate([rel_pos_np, rel_vel_np])
                    feature_kj = np.concatenate([-rel_pos_np, -rel_vel_np])
                elif feature_type == 'ga':
                    relative_pos_ga = pos_j_ga - pos_k_ga; relative_vel_ga = vel_j_ga - vel_k_ga
                    rel_pos_coeffs = np.array([relative_pos_ga[e1], relative_pos_ga[e2], relative_pos_ga[e3]])
                    rel_vel_coeffs = np.array([relative_vel_ga[e1], relative_vel_ga[e2], relative_vel_ga[e3]])
                    feature = np.concatenate([b_j_coeffs, b_k_coeffs, rel_pos_coeffs, rel_vel_coeffs])
                    feature_kj = np.concatenate([b_k_coeffs, b_j_coeffs, -rel_pos_coeffs, -rel_vel_coeffs])
                elif feature_type == 'hybrid':
                    rel_pos_np = pos_j_np - pos_k_np; rel_vel_np = vel_j_np - vel_k_np
                    feature = np.concatenate([b_j_coeffs, b_k_coeffs, rel_pos_np, rel_vel_np])
                    feature_kj = np.concatenate([b_k_coeffs, b_j_coeffs, -rel_pos_np, -rel_vel_np])
                feature_tensor[t, j, k, :] = feature; feature_tensor[t, k, j, :] = feature_kj
    X_list, y_list = [], []
    for i in range(num_steps - sequence_length - prediction_horizon + 1):
        start_idx = i; end_idx = i + sequence_length
        label_idx = end_idx + prediction_horizon - 1
        for j in range(num_sats):
            for k in range(j + 1, num_sats):
                X_list.append(feature_tensor[start_idx:end_idx, j, k, :])
                y_list.append(connectivity[label_idx, j, k])
    return X_list, y_list

def prepare_data_for_ablation(constellations, feature_types, sequence_length=5, prediction_horizon=1):
    datasets = {}
    for ft in feature_types:
        print(f"\n--- Chuẩn bị dữ liệu cho feature type: '{ft}' ---")
        all_X, all_y = [], []
        for group in constellations:
            try:
                data = np.load(f'sim_data_{group}.npz', allow_pickle=True)
                states = data['states']; connectivity = data['connectivity']
                states[:, 0] = states[:, 0] - states[0, 0]
                X_list, y_list = process_single_constellation(states, connectivity, ft, sequence_length, prediction_horizon)
                all_X.extend(X_list); all_y.extend(y_list)
            except FileNotFoundError:
                print(f"Cảnh báo: Không tìm thấy file sim_data_{group}.npz")
        datasets[ft] = {
            'X': np.array(all_X, dtype=np.float32),
            'y': np.array(all_y, dtype=np.float32).reshape(-1, 1)
        }
    return datasets

def train_for_ablation(model, X_train, y_train, X_test, y_test):
    # Hàm huấn luyện rút gọn, không cần theo dõi validation
    scaler = StandardScaler()
    X_train_shape = X_train.shape; X_train_2d = X_train.reshape(-1, X_train_shape[-1])
    scaler.fit(X_train_2d); X_train_scaled_2d = scaler.transform(X_train_2d)
    X_train = X_train_scaled_2d.reshape(X_train_shape)
    X_test_shape = X_test.shape; X_test_2d = X_test.reshape(-1, X_test_shape[-1])
    X_test_scaled_2d = scaler.transform(X_test_2d)
    X_test = X_test_scaled_2d.reshape(X_test_shape)
    X_train_tensor = torch.from_numpy(X_train); y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test); y_test_tensor = torch.from_numpy(y_test)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
    num_neg = np.sum(y_train == 0); num_pos = np.sum(y_train == 1)
    pos_weight = torch.tensor([(num_neg / num_pos if num_pos > 0 else 1.0)], dtype=torch.float32)
    criterion = nn.BCELoss(weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 15 # Huấn luyện ít hơn để tiết kiệm thời gian, 15 là đủ cho ablation
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"  Epoch [{epoch+1}/{num_epochs}] done.")
    with torch.no_grad():
        y_predicted_cls = model(X_test_tensor).round().numpy()
        return f1_score(y_test_tensor.numpy(), y_predicted_cls)

def main():
    output_dir = 'paper_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    constellations = ['iridium', 'starlink', 'oneweb']
    horizon_steps = 5; seq_len = 5
    feature_types_for_ablation = ['bivector_only', 'relative_only', 'ga', 'hybrid']
    
    # Chuẩn bị tất cả các bộ dữ liệu cần thiết
    datasets = prepare_data_for_ablation(constellations, feature_types_for_ablation, seq_len, horizon_steps)

    ablation_results = {}
    
    for ft in feature_types_for_ablation:
        print(f"\n--- Bắt đầu Huấn luyện cho Ablation Study: LSTM ({ft}) ---")
        X, y = datasets[ft]['X'], datasets[ft]['y']
        
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Khởi tạo mô hình và huấn luyện
        model = LSTM_Predictor(input_size=X.shape[2])
        f1 = train_for_ablation(model, X_train, y_train, X_test, y_test)
        
        # Lưu kết quả
        ablation_results[ft] = f1
    
    # --- Vẽ Figure 4 - Ablation Study ---
    print("\n--- Generating Figure 4: Feature Ablation Study ---")
    
    # Định dạng lại tên cho đẹp
    formatted_names = {
        'bivector_only': 'Bivector Only',
        'relative_only': 'Relative Vec. Only',
        'ga': 'GA (Combined)',
        'hybrid': 'Hybrid (Combined)'
    }
    
    names = [formatted_names[ft] for ft in feature_types_for_ablation]
    scores = [ablation_results[ft] for ft in feature_types_for_ablation]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['#8ecae6', '#219ebc', '#ffb703', '#fb8500']
    bars = ax.bar(names, scores, color=colors, edgecolor='black')
    
    ax.set_ylabel('F1 Score on Test Set', fontsize=14)
    ax.set_title('Ablation Study of Feature Components for LSTM Model (5-min Horizon)', fontsize=18, pad=20, weight='bold')
    ax.set_ylim(bottom=min(scores) * 0.99, top=max(scores) * 1.01)
    ax.tick_params(axis='x', labelsize=12, rotation=15)
    ax.tick_params(axis='y', labelsize=12)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center', fontsize=12, weight='bold')
        
    plt.tight_layout()
    fig_filename = os.path.join(output_dir, 'fig_ablation_study.png')
    plt.savefig(fig_filename, dpi=600)
    print(f"Saved Figure 4 to '{fig_filename}'")

if __name__ == '__main__':
    main()
