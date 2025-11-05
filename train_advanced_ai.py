# train_advanced_ai.py

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

# ===================================================================
# 1. ĐỊNH NGHĨA CÁC KIẾN TRÚC AI
# ===================================================================

# MLP (Giữ nguyên)
class MLP_Predictor(nn.Module):
    def __init__(self, input_size):
        super(MLP_Predictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)

# LSTM
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
def prepare_data_sequences(states, connectivity, feature_type='vector', sequence_length=5, prediction_horizon=1):
    print(f"--- Chuẩn bị dữ liệu chuỗi (length={sequence_length}) với features '{feature_type}' ---")
    
    num_steps, num_sats, _ = connectivity.shape
    
    # Tạo state tensor (pos, vel)
    unique_sat_ids = np.unique(states[:, 1])
    sat_id_to_idx = {int(sat_id): i for i, sat_id in enumerate(unique_sat_ids)}
    state_tensor = np.zeros((num_steps, num_sats, 6))
    for row in states:
        step, sat_id, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z = row
        state_tensor[int(step), sat_id_to_idx[int(sat_id)], :] = [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
    
    # Tạo feature tensor dựa trên feature_type
    if feature_type == 'vector':
        # Input là state_j và state_k ghép lại
        feature_dim = 12
        feature_tensor = np.zeros((num_steps, num_sats, num_sats, feature_dim))
        for j in range(num_sats):
            for k in range(j + 1, num_sats):
                feature = np.concatenate([state_tensor[:, j, :], state_tensor[:, k, :]], axis=1)
                feature_tensor[:, j, k, :] = feature
                feature_tensor[:, k, j, :] = np.concatenate([state_tensor[:, k, :], state_tensor[:, j, :]], axis=1)

    elif feature_type == 'ga':
        layout_g3, blades_g3 = clifford.Cl(3)
        e1, e2, e3 = blades_g3['e1'], blades_g3['e2'], blades_g3['e3']
        feature_dim = 12
        feature_tensor = np.zeros((num_steps, num_sats, num_sats, feature_dim))
        for j in range(num_sats):
            for k in range(j + 1, num_sats):
                # Tính đặc trưng GA cho tất cả các bước thời gian cùng lúc
                pos_j_ga = state_tensor[:, j, 0:1]*e1 + state_tensor[:, j, 1:2]*e2 + state_tensor[:, j, 2:3]*e3
                vel_j_ga = state_tensor[:, j, 3:4]*e1 + state_tensor[:, j, 4:5]*e2 + state_tensor[:, j, 5:6]*e3
                pos_k_ga = state_tensor[:, k, 0:1]*e1 + state_tensor[:, k, 1:2]*e2 + state_tensor[:, k, 2:3]*e3
                vel_k_ga = state_tensor[:, k, 3:4]*e1 + state_tensor[:, k, 4:5]*e2 + state_tensor[:, k, 5:6]*e3
                
                bivector_j = pos_j_ga ^ vel_j_ga
                bivector_k = pos_k_ga ^ vel_k_ga
                relative_pos = pos_j_ga - pos_k_ga
                relative_vel = vel_j_ga - vel_k_ga

                b_j_coeffs = np.hstack([bivector_j[e1^e2], bivector_j[e1^e3], bivector_j[e2^e3]])
                b_k_coeffs = np.hstack([bivector_k[e1^e2], bivector_k[e1^e3], bivector_k[e2^e3]])
                rel_pos_coeffs = np.hstack([relative_pos[e1], relative_pos[e2], relative_pos[e3]])
                rel_vel_coeffs = np.hstack([relative_vel[e1], relative_vel[e2], relative_vel[e3]])
                
                feature = np.concatenate([b_j_coeffs, b_k_coeffs, rel_pos_coeffs, rel_vel_coeffs], axis=1)
                feature_tensor[:, j, k, :] = feature
                # Đối với cặp (k,j), pos và vel tương đối sẽ đổi dấu
                feature_kj = np.concatenate([b_k_coeffs, b_j_coeffs, -rel_pos_coeffs, -rel_vel_coeffs], axis=1)
                feature_tensor[:, k, j, :] = feature_kj

    # Tạo các chuỗi dữ liệu
    X_list, y_list = [], []
    for i in range(num_steps - sequence_length - prediction_horizon + 1):
        start_idx = i
        end_idx = i + sequence_length
        label_idx = end_idx + prediction_horizon - 1
        
        for j in range(num_sats):
            for k in range(j + 1, num_sats):
                sequence = feature_tensor[start_idx:end_idx, j, k, :]
                label = connectivity[label_idx, j, k]
                X_list.append(sequence)
                y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    print(f"Đã tạo {len(X)} mẫu chuỗi. Shape X: {X.shape}, Shape y: {y.shape}")
    return X, y

# ===================================================================
# 3. HÀM HUẤN LUYỆN VÀ ĐÁNH GIÁ
# ===================================================================
def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    print(f"\n--- Bắt đầu Huấn luyện: {model_name} ---")
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    # Reshape để scaler hoạt động trên dữ liệu chuỗi
    X_train_shape = X_train.shape
    X_train_2d = X_train.reshape(-1, X_train_shape[-1])
    scaler.fit(X_train_2d)
    X_train_scaled_2d = scaler.transform(X_train_2d)
    X_train = X_train_scaled_2d.reshape(X_train_shape)

    X_test_shape = X_test.shape
    X_test_2d = X_test.reshape(-1, X_test_shape[-1])
    X_test_scaled_2d = scaler.transform(X_test_2d)
    X_test = X_test_scaled_2d.reshape(X_test_shape)

    # Nếu là MLP, flatten chuỗi
    if isinstance(model, MLP_Predictor):
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
    
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
    
    criterion = nn.BCELoss(weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5 # Giảm số epoch cho thử nghiệm nhanh
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Đánh giá
    with torch.no_grad():
        y_predicted = model(X_test_tensor)
        y_predicted_cls = y_predicted.round()
        f1 = f1_score(y_test_tensor, y_predicted_cls)
        auc = roc_auc_score(y_test_tensor, y_predicted)
        return {'F1 Score': f1, 'AUC-ROC': auc}

# ===================================================================
# 4. HÀM CHÍNH
# ===================================================================
def main():
    # Tải và kết hợp dữ liệu
    print("--- Tải và kết hợp dữ liệu từ các chòm vệ tinh ---")
    all_states, all_connectivity = [], []
    for group in ['iridium', 'starlink', 'oneweb']:
        try:
            data = np.load(f'sim_data_{group}.npz')
            print(f"Đã tải {data['states'].shape[0]} state entries từ {group}.")
            # Cần cập nhật lại step index để nối dữ liệu
            current_steps = len(all_states) // (data['states'].shape[0] // data['connectivity'].shape[0]) if len(all_states) > 0 else 0
            states_data = data['states']
            states_data[:, 0] += current_steps
            all_states.append(states_data)
            all_connectivity.append(data['connectivity'])
        except FileNotFoundError:
            print(f"Cảnh báo: Không tìm thấy file sim_data_{group}.npz")
            
    if not all_states:
        print("Lỗi: Không có dữ liệu để xử lý.")
        return
        
    states = np.vstack(all_states)
    connectivity = np.vstack(all_connectivity)

    # Chuẩn bị 2 bộ dữ liệu (vector và ga)
    X_vec, y_vec = prepare_data_sequences(states, connectivity, feature_type='vector')
    X_ga, y_ga = prepare_data_sequences(states, connectivity, feature_type='ga')

    # Chia dữ liệu
    X_vec_train, X_vec_test, y_train, y_test = train_test_split(X_vec, y_vec, test_size=0.3, random_state=42, stratify=y_vec)
    X_ga_train, X_ga_test, _, _ = train_test_split(X_ga, y_ga, test_size=0.3, random_state=42, stratify=y_ga)

    # Chạy thực nghiệm
    results = []
    seq_len, feat_dim = X_vec.shape[1], X_vec.shape[2]
    
    # 1. MLP Baseline (dùng feature Vector)
    model_mlp = MLP_Predictor(input_size=seq_len * feat_dim)
    res_mlp = train_and_evaluate(model_mlp, X_vec_train, y_train, X_vec_test, y_test, "MLP Baseline (Vector)")
    results.append({'Model': 'MLP', 'Features': 'Vector', **res_mlp})

    # 2. LSTM (dùng feature Vector)
    model_lstm_vec = LSTM_Predictor(input_size=feat_dim)
    res_lstm_vec = train_and_evaluate(model_lstm_vec, X_vec_train, y_train, X_vec_test, y_test, "LSTM (Vector)")
    results.append({'Model': 'LSTM', 'Features': 'Vector', **res_lstm_vec})
    
    # 3. LSTM (dùng feature GA)
    model_lstm_ga = LSTM_Predictor(input_size=feat_dim)
    res_lstm_ga = train_and_evaluate(model_lstm_ga, X_ga_train, y_train, X_ga_test, y_test, "LSTM (GA)")
    results.append({'Model': 'LSTM', 'Features': 'GA', **res_lstm_ga})
    
    # In kết quả cuối cùng
    print("\n\n--- BẢNG KẾT QUẢ SO SÁNH CUỐI CÙNG ---")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
