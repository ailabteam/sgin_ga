# train_final_comparison.py
# Phiên bản thực nghiệm cuối cùng, so sánh Vector, GA, và Hybrid features trên mô hình LSTM
# cho bài toán dự đoán dài hạn (5 phút).

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import clifford
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
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
        # x shape: (batch_size, seq_length, input_size)
        out, _ = self.lstm(x)
        # Lấy output của bước thời gian cuối cùng
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

    # Khởi tạo GA một lần để tái sử dụng
    layout_g3, blades_g3 = clifford.Cl(3)
    e1, e2, e3 = blades_g3['e1'], blades_g3['e2'], blades_g3['e3']

    # Lặp qua từng bước thời gian để tính toán
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
                        # Vector tương đối trực tiếp từ numpy
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

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    print(f"\n--- Bắt đầu Huấn luyện: {model_name} ---")
    start_time = time.time()
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
    
    X_train_tensor = torch.from_numpy(X_train); y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test); y_test_tensor = torch.from_numpy(y_test)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
    
    num_neg = np.sum(y_train == 0); num_pos = np.sum(y_train == 1)
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
    criterion = nn.BCELoss(weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    end_time = time.time()
    training_time = end_time - start_time

    with torch.no_grad():
        y_predicted = model(X_test_tensor)
        y_predicted_cls = y_predicted.round()
        f1 = f1_score(y_test_tensor, y_predicted_cls)
        auc = roc_auc_score(y_test_tensor, y_predicted)
        return {'F1 Score': f1, 'AUC-ROC': auc, 'Training Time (s)': training_time}

def main():
    constellations = ['iridium', 'starlink', 'oneweb']
    horizon_steps = 5 # Dự đoán dài hạn 5 phút
    seq_len = 5
    
    X_vec, y_vec = prepare_combined_data(constellations, 'vector', seq_len, horizon_steps)
    X_ga, y_ga = prepare_combined_data(constellations, 'ga', seq_len, horizon_steps)
    X_hyb, y_hyb = prepare_combined_data(constellations, 'hybrid', seq_len, horizon_steps)
    
    X_vec_train, X_vec_test, y_train, y_test = train_test_split(X_vec, y_vec, test_size=0.3, random_state=42, stratify=y_vec)
    X_ga_train, X_ga_test, _, _ = train_test_split(X_ga, y_ga, test_size=0.3, random_state=42, stratify=y_ga)
    X_hyb_train, X_hyb_test, _, _ = train_test_split(X_hyb, y_hyb, test_size=0.3, random_state=42, stratify=y_hyb)

    results = []
    feat_dim = X_vec.shape[2]
    
    # Thêm MLP để so sánh
    model_mlp = MLP_Predictor(input_size=seq_len * feat_dim)
    res_mlp = train_and_evaluate(model_mlp, X_vec_train, y_train, X_vec_test, y_test, "MLP (Vector)")
    results.append({'Model': 'MLP', 'Features': 'Vector', **res_mlp})

    model_lstm_vec = LSTM_Predictor(input_size=feat_dim)
    res_lstm_vec = train_and_evaluate(model_lstm_vec, X_vec_train, y_train, X_vec_test, y_test, "LSTM (Vector)")
    results.append({'Model': 'LSTM', 'Features': 'Vector', **res_lstm_vec})
    
    model_lstm_ga = LSTM_Predictor(input_size=feat_dim)
    res_lstm_ga = train_and_evaluate(model_lstm_ga, X_ga_train, y_train, X_ga_test, y_test, "LSTM (GA)")
    results.append({'Model': 'LSTM', 'Features': 'GA', **res_lstm_ga})
    
    model_lstm_hyb = LSTM_Predictor(input_size=feat_dim)
    res_lstm_hyb = train_and_evaluate(model_lstm_hyb, X_hyb_train, y_train, X_hyb_test, y_test, "LSTM (Hybrid)")
    results.append({'Model': 'LSTM', 'Features': 'Hybrid', **res_lstm_hyb})
    
    print("\n\n--- BẢNG KẾT QUẢ SO SÁNH CUỐI CÙNG (DỰ ĐOÁN 5 PHÚT) ---")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
