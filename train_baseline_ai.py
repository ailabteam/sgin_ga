# train_baseline_ai.py (v1.1 - Xử lý dữ liệu mất cân bằng)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class LoS_Predictor(nn.Module):
    # ... (Giữ nguyên)
    def __init__(self, input_size):
        super(LoS_Predictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

def prepare_data(states, connectivity, prediction_horizon_steps):
    # ... (Giữ nguyên)
    print(f"--- Chuẩn bị dữ liệu cho AI (Dự đoán trước {prediction_horizon_steps} bước) ---")
    num_steps, num_sats, _ = connectivity.shape
    X_list = []
    y_list = []
    unique_sat_ids = np.unique(states[:, 1])
    sat_id_to_idx = {int(sat_id): i for i, sat_id in enumerate(unique_sat_ids)}
    state_tensor = np.zeros((num_steps, num_sats, 6))
    for row in states:
        step = int(row[0])
        sat_idx = sat_id_to_idx[int(row[1])]
        state_tensor[step, sat_idx, :] = row[2:]
    for i in range(num_steps - prediction_horizon_steps):
        current_step = i
        future_step = i + prediction_horizon_steps
        for j in range(num_sats):
            for k in range(j + 1, num_sats):
                state_j = state_tensor[current_step, j, :]
                state_k = state_tensor[current_step, k, :]
                input_vector = np.concatenate([state_j, state_k])
                X_list.append(input_vector)
                label = connectivity[future_step, j, k]
                y_list.append(label)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    print(f"Đã tạo thành công {len(X)} mẫu dữ liệu.")
    print(f"Kích thước Input X: {X.shape}")
    print(f"Kích thước Label y: {y.shape}")
    return X, y

def main():
    # --- Tải dữ liệu ---
    data = np.load('simulation_data.npz')
    states = data['states']; connectivity = data['connectivity']

    # --- Chuẩn bị dữ liệu ---
    HORIZON_SECONDS = 60; TIME_STEP = 60
    horizon_steps = HORIZON_SECONDS // TIME_STEP
    X, y = prepare_data(states, connectivity, horizon_steps)

    # --- Chia dữ liệu ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Chuyển dữ liệu sang PyTorch Tensors ---
    X_train_tensor = torch.from_numpy(X_train); y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test); y_test_tensor = torch.from_numpy(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True) # Tăng batch size

    # ===================================================================
    # SỬA LỖI: TÍNH TOÁN VÀ ÁP DỤNG TRỌNG SỐ CHO HÀM MẤT MÁT
    # ===================================================================
    print("\n--- Xử lý Dữ liệu Mất cân bằng ---")
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
    print(f"Tỷ lệ Neg/Pos trong tập train: {pos_weight.item():.2f}")
    print("Sử dụng trọng số này để 'phạt' nặng hơn khi đoán sai lớp 'Có LoS'.")
    # ===================================================================

    # --- Khởi tạo và Huấn luyện mô hình ---
    print("\n--- Bắt đầu Huấn luyện Mô hình AI Baseline (v1.1) ---")
    model = LoS_Predictor(X_train.shape[1])
    
    # Truyền `pos_weight` vào hàm mất mát
    criterion = nn.BCELoss(weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_loader):
            outputs = model(features)
            
            # Cần tính loss cho từng mẫu với trọng số tương ứng
            # BCELoss với `weight` đã làm điều này cho ta
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # --- Đánh giá mô hình ---
    print("\n--- Đánh giá trên tập Test ---")
    with torch.no_grad():
        y_predicted = model(X_test_tensor)
        y_predicted_cls = y_predicted.round()
        
        print(f'Accuracy: {accuracy_score(y_test_tensor, y_predicted_cls):.4f}')
        print(f'Precision: {precision_score(y_test_tensor, y_predicted_cls):.4f}')
        print(f'Recall: {recall_score(y_test_tensor, y_predicted_cls):.4f}')
        print(f'F1 Score: {f1_score(y_test_tensor, y_predicted_cls):.4f}')
        print(f'AUC-ROC: {roc_auc_score(y_test_tensor, y_predicted):.4f}') # Thêm chỉ số AUC

if __name__ == '__main__':
    main()
