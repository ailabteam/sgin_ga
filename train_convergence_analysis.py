# train_convergence_analysis.py
# Thực nghiệm cuối cùng: Phân tích sự hội tụ của các mô hình MLP/LSTM
# với các bộ đặc trưng Vector, GA, và Hybrid qua 30 epochs.

import numpy as np
import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from sklearn.preprocessing import StandardScaler
import clifford
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import time

# ===================================================================
# 1. ĐỊNH NGHĨA CÁC KIẾN TRÚC AI
# ===================================================================
class MLP_Predictor(nn.Module):
    def __init__(self, input_size):
        super(MLP_Predictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)

class LSTM_Predictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTM_Predictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# ===================================================================
# 2. CÁC HÀM CHUẨN BỊ DỮ LIỆU
# ===================================================================
def process_single_constellation(states, connectivity, feature_type='vector', sequence_length=5, prediction_horizon=1):
    num_steps, num_sats, _ = connectivity.shape
    unique_sat_ids = np.unique(states[:, 1])
    sat_id_to_idx = {int(sat_id): i for i, sat_id in enumerate(unique_sat_ids)}

    if num_sats != len(unique_sat_ids):
        print(f"  - Cảnh báo: Số vệ tinh không khớp. Connectivity: {num_sats}, States: {len(unique_sat_ids)}. Bỏ qua chòm này.")
        return [], []

    state_tensor = np.zeros((num_steps, num_sats, 6))
    for row in states:
        step, sat_id, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z = row
        if int(sat_id) in sat_id_to_idx:
            state_tensor[int(step), sat_id_to_idx[int(sat_id)], :] = [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]

    feature_dim = 12
    feature_tensor = np.zeros((num_steps, num_sats, num_sats, feature_dim))

    layout_g3, blades_g3 = clifford.Cl(3)
    e1, e2, e3 = blades_g3['e1'], blades_g3['e2'], blades_g3['e3']

    for t in range(num_steps):
        for j in range(num_sats):
            for k in range(j + 1, num_sats):
                state_j_np = state_tensor[t, j, :]
                state_k_np = state_tensor[t, k, :]

                if feature_type == 'vector':
                    feature = np.concatenate([state_j_np, state_k_np])
                    feature_kj = np.concatenate([state_k_np, state_j_np])
                elif feature_type in ['ga', 'hybrid']:
                    pos_j_np, vel_j_np = state_j_np[:3], state_j_np[3:]
                    pos_k_np, vel_k_np = state_k_np[:3], state_k_np[3:]

                    pos_j_ga = pos_j_np[0]*e1 + pos_j_np[1]*e2 + pos_j_np[2]*e3
                    vel_j_ga = vel_j_np[0]*e1 + vel_j_np[1]*e2 + vel_j_np[2]*e3
                    pos_k_ga = pos_k_np[0]*e1 + pos_k_np[1]*e2 + pos_k_np[2]*e3
                    vel_k_ga = vel_k_np[0]*e1 + vel_k_np[1]*e2 + vel_k_np[2]*e3

                    bivector_j = pos_j_ga ^ vel_j_ga
                    bivector_k = pos_k_ga ^ vel_k_ga

                    b_j_coeffs = np.array([bivector_j[e1^e2], bivector_j[e1^e3], bivector_j[e2^e3]])
                    b_k_coeffs = np.array([bivector_k[e1^e2], bivector_k[e1^e3], bivector_k[e2^e3]])

                    relative_pos_ga = pos_j_ga - pos_k_ga
                    relative_vel_ga = vel_j_ga - vel_k_ga
                    rel_pos_coeffs = np.array([relative_pos_ga[e1], relative_pos_ga[e2], relative_pos_ga[e3]])
                    rel_vel_coeffs = np.array([relative_vel_ga[e1], relative_vel_ga[e2], relative_vel_ga[e3]])

                    if feature_type == 'ga':
                        feature = np.concatenate([b_j_coeffs, b_k_coeffs, rel_pos_coeffs, rel_vel_coeffs])
                        feature_kj = np.concatenate([b_k_coeffs, b_j_coeffs, -rel_pos_coeffs, -rel_vel_coeffs])
                    elif feature_type == 'hybrid':
                        rel_pos_np = pos_j_np - pos_k_np
                        rel_vel_np = vel_j_np - vel_k_np
                        feature = np.concatenate([b_j_coeffs, b_k_coeffs, rel_pos_np, rel_vel_np])
                        feature_kj = np.concatenate([b_k_coeffs, b_j_coeffs, -rel_pos_np, -rel_vel_np])

                feature_tensor[t, j, k, :] = feature
                feature_tensor[t, k, j, :] = feature_kj

    X_list, y_list = [], []
    for i in range(num_steps - sequence_length - prediction_horizon + 1):
        start_idx = i; end_idx = i + sequence_length
        label_idx = end_idx + prediction_horizon - 1
        for j in range(num_sats):
            for k in range(j + 1, num_sats):
                X_list.append(feature_tensor[start_idx:end_idx, j, k, :])
                y_list.append(connectivity[label_idx, j, k])
    return X_list, y_list

def prepare_combined_data(constellations, feature_type, sequence_length=5, prediction_horizon=1):
    print(f"\n--- Chuẩn bị dữ liệu kết hợp với features '{feature_type}' ---")
    all_X, all_y = [], []
    for group in constellations:
        try:
            print(f"Đang xử lý {group}...")
            data = np.load(f'sim_data_{group}.npz', allow_pickle=True)
            states = data['states']; connectivity = data['connectivity']
            states[:, 0] = states[:, 0] - states[0, 0]
            X_list, y_list = process_single_constellation(states, connectivity, feature_type, sequence_length, prediction_horizon)
            all_X.extend(X_list); all_y.extend(y_list)
        except FileNotFoundError:
            print(f"Cảnh báo: Không tìm thấy file sim_data_{group}.npz")

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32).reshape(-1, 1)
    print(f"Đã tạo {len(X)} mẫu chuỗi tổng hợp. Shape X: {X.shape}, Shape y: {y.shape}")
    return X, y

# ===================================================================
# 3. HÀM HUẤN LUYỆN VÀ ĐÁNH GIÁ (NÂNG CẤP)
# ===================================================================

def train_and_evaluate_convergence(model, X_train, y_train, X_test, y_test, model_name, num_epochs=30):
    print(f"\n--- Bắt đầu Phân tích Hội tụ: {model_name} ({num_epochs} epochs) ---")
    start_time = time.time()
    
    # Chuẩn hóa dữ liệu (không đổi)
    scaler = StandardScaler()
    X_train_shape = X_train.shape; X_train_2d = X_train.reshape(-1, X_train_shape[-1])
    scaler.fit(X_train_2d); X_train_scaled_2d = scaler.transform(X_train_2d)
    X_train = X_train_scaled_2d.reshape(X_train_shape)
    X_test_shape = X_test.shape; X_test_2d = X_test.reshape(-1, X_test_shape[-1])
    X_test_scaled_2d = scaler.transform(X_test_2d)
    X_test = X_test_scaled_2d.reshape(X_test_shape)

    if isinstance(model, MLP_Predictor):
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test) # Dùng cho đánh giá cuối
    
    # Tách tập validation (không đổi)
    full_train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(dataset=train_subset, batch_size=512, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, batch_size=1024)

    # Vòng lặp huấn luyện và theo dõi (không đổi)
    y_train_new = y_train_tensor[train_subset.indices]
    num_neg = torch.sum(y_train_new == 0); num_pos = torch.sum(y_train_new == 1)
    if num_pos == 0: num_pos = 1
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
    criterion = nn.BCELoss(weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history = {'val_f1': []}

    for epoch in range(num_epochs):
        model.train()
        for features, labels in train_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                val_preds.append(outputs.round())
                val_labels.append(labels)
        
        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        history['val_f1'].append(val_f1)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val F1: {val_f1:.4f}')
    
    training_time = time.time() - start_time
    
    # CẬP NHẬT: Đánh giá cuối cùng với đầy đủ chỉ số
    model.eval()
    with torch.no_grad():
        y_predicted_proba = model(X_test_tensor)
        y_predicted_cls = y_predicted_proba.round().numpy()
        
        # Chuyển y_test_tensor sang numpy để dùng với scikit-learn
        y_test_np = y_test_tensor.numpy()
        
        final_f1 = f1_score(y_test_np, y_predicted_cls)
        final_auc = roc_auc_score(y_test_np, y_predicted_proba.numpy())
        final_precision = precision_score(y_test_np, y_predicted_cls, zero_division=0)
        final_recall = recall_score(y_test_np, y_predicted_cls, zero_division=0)
    
    return {'F1 Score': final_f1, 'AUC-ROC': final_auc, 
            'Precision': final_precision, 'Recall': final_recall, 
            'Training Time (s)': training_time, 'History': history}

# ===================================================================
# 4. HÀM MAIN
# ===================================================================
def main():
    # CẬP NHẬT: Tạo thư mục output
    output_dir = 'paper_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    start_total_time = time.time()
    # Các thiết lập không đổi
    constellations = ['iridium', 'starlink', 'oneweb']
    horizon_steps = 5
    seq_len = 5
    NUM_EPOCHS = 30

    # Chuẩn bị 3 bộ dữ liệu (không đổi)
    X_vec, y_vec = prepare_combined_data(constellations, 'vector', seq_len, horizon_steps)
    X_ga, y_ga = prepare_combined_data(constellations, 'ga', seq_len, horizon_steps)
    X_hyb, y_hyb = prepare_combined_data(constellations, 'hybrid', seq_len, horizon_steps)
    
    # Chia dữ liệu (không đổi)
    X_vec_train, X_vec_test, y_train, y_test = train_test_split(X_vec, y_vec, test_size=0.3, random_state=42, stratify=y_vec)
    X_ga_train, X_ga_test, _, _ = train_test_split(X_ga, y_ga, test_size=0.3, random_state=42, stratify=y_ga)
    X_hyb_train, X_hyb_test, _, _ = train_test_split(X_hyb, y_hyb, test_size=0.3, random_state=42, stratify=y_hyb)

    # Chạy thực nghiệm (không đổi)
    results = []
    all_histories = {}
    feat_dim = X_vec.shape[2]
    
    # Chạy MLP
    model_mlp = MLP_Predictor(input_size=seq_len * feat_dim)
    res_mlp = train_and_evaluate_convergence(model_mlp, X_vec_train, y_train, X_vec_test, y_test, "MLP (Vector)", NUM_EPOCHS)
    all_histories['MLP (Vector)'] = res_mlp.pop('History')
    results.append({'Model': 'MLP', 'Features': 'Vector', **res_mlp})

    # Chạy LSTM (Vector)
    model_lstm_vec = LSTM_Predictor(input_size=feat_dim)
    res_lstm_vec = train_and_evaluate_convergence(model_lstm_vec, X_vec_train, y_train, X_vec_test, y_test, "LSTM (Vector)", NUM_EPOCHS)
    all_histories['LSTM (Vector)'] = res_lstm_vec.pop('History')
    results.append({'Model': 'LSTM', 'Features': 'Vector', **res_lstm_vec})
    
    # Chạy LSTM (GA)
    model_lstm_ga = LSTM_Predictor(input_size=feat_dim)
    res_lstm_ga = train_and_evaluate_convergence(model_lstm_ga, X_ga_train, y_train, X_ga_test, y_test, "LSTM (GA)", NUM_EPOCHS)
    all_histories['LSTM (GA)'] = res_lstm_ga.pop('History')
    results.append({'Model': 'LSTM', 'Features': 'GA', **res_lstm_ga})
    
    # Chạy LSTM (Hybrid)
    model_lstm_hyb = LSTM_Predictor(input_size=feat_dim)
    res_lstm_hyb = train_and_evaluate_convergence(model_lstm_hyb, X_hyb_train, y_train, X_hyb_test, y_test, "LSTM (Hybrid)", NUM_EPOCHS)
    all_histories['LSTM (Hybrid)'] = res_lstm_hyb.pop('History')
    results.append({'Model': 'LSTM', 'Features': 'Hybrid', **res_lstm_hyb})
    
    # CẬP NHẬT: In bảng kết quả đầy đủ và sắp xếp cột
    print("\n\n--- BẢNG KẾT QUẢ CUỐI CÙNG (SAU 30 EPOCHS) ---")
    df = pd.DataFrame(results)
    df = df[['Model', 'Features', 'F1 Score', 'AUC-ROC', 'Precision', 'Recall', 'Training Time (s)']]
    print(df.to_string(index=False))

    # CẬP NHẬT: Vẽ và lưu biểu đồ với DPI 600 vào đúng thư mục
    print("\n--- Đang tạo biểu đồ hội tụ... ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, history in all_histories.items():
        ax.plot(range(1, NUM_EPOCHS + 1), history['val_f1'], marker='o', linestyle='-', markersize=4, label=name)
    
    ax.set_title('Convergence Analysis: F1 Score on Validation Set over Epochs', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation F1 Score', fontsize=12)
    ax.set_xticks(np.arange(0, NUM_EPOCHS + 1, 2))
    plt.xticks(rotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    
    fig_filename = os.path.join(output_dir, 'fig_convergence_analysis.png')
    plt.savefig(fig_filename, dpi=600)
    print(f"Đã lưu biểu đồ vào file '{fig_filename}'.")

    end_total_time = time.time()
    print(f"\nToàn bộ quá trình thực nghiệm hoàn tất sau {(end_total_time - start_total_time) / 60:.2f} phút.")

if __name__ == '__main__':
    main()

