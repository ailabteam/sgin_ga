# train_ga_ai.py (v1.1 - Thêm bước Chuẩn hóa Dữ liệu)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler # <<< THÊM VÀO
import clifford
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

# --- Các lớp và hàm không đổi ---
class LoS_Predictor(nn.Module):
    # ... (Giữ nguyên)
    def __init__(self, input_size):
        super(LoS_Predictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)

def prepare_data_ga(states, connectivity, prediction_horizon_steps):
    # ... (Giữ nguyên)
    print(f"--- Chuẩn bị dữ liệu cho AI (với Đặc trưng Geometric Algebra) ---")
    layout_g3, blades_g3 = clifford.Cl(3)
    e1, e2, e3 = blades_g3['e1'], blades_g3['e2'], blades_g3['e3']
    num_steps, num_sats, _ = connectivity.shape
    X_list = []; y_list = []
    unique_sat_ids = np.unique(states[:, 1])
    sat_id_to_idx = {int(sat_id): i for i, sat_id in enumerate(unique_sat_ids)}
    state_tensor = np.zeros((num_steps, num_sats, 6))
    for row in states:
        step = int(row[0]); sat_idx = sat_id_to_idx[int(row[1])]
        state_tensor[step, sat_idx, :] = row[2:]
    for i in range(num_steps - prediction_horizon_steps):
        current_step = i; future_step = i + prediction_horizon_steps
        for j in range(num_sats):
            for k in range(j + 1, num_sats):
                state_j_np = state_tensor[current_step, j, :]
                state_k_np = state_tensor[current_step, k, :]
                pos_j_np, vel_j_np = state_j_np[:3], state_j_np[3:]
                pos_k_np, vel_k_np = state_k_np[:3], state_k_np[3:]
                pos_j_ga = pos_j_np[0]*e1 + pos_j_np[1]*e2 + pos_j_np[2]*e3
                vel_j_ga = vel_j_np[0]*e1 + vel_j_np[1]*e2 + vel_j_np[2]*e3
                pos_k_ga = pos_k_np[0]*e1 + pos_k_np[1]*e2 + pos_k_np[2]*e3
                vel_k_ga = vel_k_np[0]*e1 + vel_k_np[1]*e2 + vel_k_np[2]*e3
                bivector_j = pos_j_ga ^ vel_j_ga; bivector_k = pos_k_ga ^ vel_k_ga
                relative_pos = pos_j_ga - pos_k_ga; relative_vel = vel_j_ga - vel_k_ga
                b_j_coeffs = np.array([bivector_j[e1^e2], bivector_j[e1^e3], bivector_j[e2^e3]])
                b_k_coeffs = np.array([bivector_k[e1^e2], bivector_k[e1^e3], bivector_k[e2^e3]])
                rel_pos_coeffs = np.array([relative_pos[e1], relative_pos[e2], relative_pos[e3]])
                rel_vel_coeffs = np.array([relative_vel[e1], relative_vel[e2], relative_vel[e3]])
                input_vector = np.concatenate([b_j_coeffs, b_k_coeffs, rel_pos_coeffs, rel_vel_coeffs])
                X_list.append(input_vector)
                y_list.append(connectivity[future_step, j, k])
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    print(f"Đã tạo thành công {len(X)} mẫu dữ liệu GA.")
    return X, y

def main():
    data = np.load('simulation_data.npz')
    states = data['states']; connectivity = data['connectivity']
    HORIZON_SECONDS = 60; TIME_STEP = 60
    horizon_steps = HORIZON_SECONDS // TIME_STEP
    X, y = prepare_data_ga(states, connectivity, horizon_steps)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # ===================================================================
    # SỬA LỖI: THÊM BƯỚC CHUẨN HÓA DỮ LIỆU
    # ===================================================================
    print("\n--- Chuẩn hóa Dữ liệu (Scaling) ---")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) # Fit và transform trên tập train
    X_test = scaler.transform(X_test)       # Chỉ transform trên tập test
    print("Đã chuẩn hóa dữ liệu đầu vào.")
    # ===================================================================

    X_train_tensor = torch.from_numpy(X_train); y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test); y_test_tensor = torch.from_numpy(y_test)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    
    num_neg = np.sum(y_train == 0); num_pos = np.sum(y_train == 1)
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)

    print("\n--- Bắt đầu Huấn luyện Mô hình AI với Đặc trưng GA (v1.1) ---")
    model = LoS_Predictor(X_train.shape[1])
    criterion = nn.BCELoss(weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        # ... (Vòng lặp huấn luyện giữ nguyên)
        for i, (features, labels) in enumerate(train_loader):
            outputs = model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("\n--- Đánh giá trên tập Test (Mô hình GA) ---")
    # ... (Phần đánh giá giữ nguyên)
    with torch.no_grad():
        y_predicted = model(X_test_tensor)
        y_predicted_cls = y_predicted.round()
        print(f'Accuracy: {accuracy_score(y_test_tensor, y_predicted_cls):.4f}')
        print(f'Precision: {precision_score(y_test_tensor, y_predicted_cls):.4f}')
        print(f'Recall: {recall_score(y_test_tensor, y_predicted_cls):.4f}')
        print(f'F1 Score: {f1_score(y_test_tensor, y_predicted_cls):.4f}')
        print(f'AUC-ROC: {roc_auc_score(y_test_tensor, y_predicted):.4f}')
        
if __name__ == '__main__':
    main()
