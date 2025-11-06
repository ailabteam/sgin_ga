# utils.py (v1.1 - Sửa lỗi NameError)
# Chứa tất cả các lớp mô hình, hàm chuẩn bị dữ liệu, và hàm huấn luyện dùng chung.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import clifford
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
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
                if feature_type == 'vector':
                    feature = np.concatenate([state_j_np, state_k_np])
                    feature_kj = np.concatenate([state_k_np, state_j_np])
                else:
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
# 3. HÀM HUẤN LUYỆN VÀ ĐÁNH GIÁ
# ===================================================================
def train_and_evaluate_convergence(model, X_train, y_train, X_test, y_test, model_name, num_epochs=30):
    print(f"\n--- Bắt đầu Phân tích Hội tụ: {model_name} ({num_epochs} epochs) ---")
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
    full_train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_size = int(0.9 * len(full_train_dataset)); val_size = len(full_train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
    train_loader = DataLoader(dataset=train_subset, batch_size=512, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, batch_size=1024)
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
                val_preds.append(outputs.round()); val_labels.append(labels)
        val_preds = torch.cat(val_preds).numpy(); val_labels = torch.cat(val_labels).numpy()
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        history['val_f1'].append(val_f1)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val F1: {val_f1:.4f}')
    training_time = time.time() - start_time
    model.eval()
    with torch.no_grad():
        y_predicted_proba = model(X_test_tensor)
        y_predicted_cls = y_predicted_proba.round().numpy()
        y_test_np = y_test_tensor.numpy()
        final_f1 = f1_score(y_test_np, y_predicted_cls)
        final_auc = roc_auc_score(y_test_np, y_predicted_proba.numpy())
        final_precision = precision_score(y_test_np, y_predicted_cls, zero_division=0)
        final_recall = recall_score(y_test_np, y_predicted_cls, zero_division=0)
    return {'F1 Score': final_f1, 'AUC-ROC': final_auc, 'Precision': final_precision, 'Recall': final_recall, 
            'Training Time (s)': training_time, 'History': history}
