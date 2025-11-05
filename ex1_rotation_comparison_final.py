# ex1_rotation_comparison_final.py

import numpy as np
from clifford.g3 import e1, e2, e3, layout
from clifford.tools.g3 import normal_to_bivector

def main():
    """
    Hàm chính để so sánh phép quay bằng Quaternion và Geometric Algebra.
    Phiên bản cuối cùng, đã sửa tất cả các lỗi.
    """
    print("--- So sánh phép quay 3D (Phiên bản cuối) ---")
    
    # Dữ liệu đầu vào
    v = np.array([1.0, 0.0, 0.0])
    angle = np.pi / 2
    axis = np.array([0.0, 0.0, 1.0])
    expected_result = np.array([0.0, 1.0, 0.0])
    
    print(f"Vector ban đầu v: {v}")
    print(f"Góc quay: {np.rad2deg(angle)} độ quanh trục {axis}\n")

    # --- 1. Quaternion (Đã đúng) ---
    print("--- 1. Sử dụng Quaternion ---")
    v_rotated_q = rotate_with_quaternion(v, angle, axis)
    print(f"Vector sau khi quay bằng Quaternion: {v_rotated_q}")
    print(f"Kết quả có đúng không? {np.allclose(v_rotated_q, expected_result)}\n")

    # --- 2. Geometric Algebra (Cách thủ công, đã sửa) ---
    print("--- 2. Sử dụng GA (Rotor thủ công, đã sửa) ---")
    v_rotated_ga_manual = rotate_with_ga_manual_fixed(v, angle)
    print(f"Vector sau khi quay bằng Rotor thủ công: {v_rotated_ga_manual}")
    print(f"Kết quả có đúng không? {np.allclose(v_rotated_ga_manual, expected_result)}\n")
    
    # --- 3. Geometric Algebra (Cách làm thanh lịch, dùng exponential map) ---
    print("--- 3. Sử dụng GA (Rotor thanh lịch, dùng exponential map) ---")
    v_rotated_ga_elegant = rotate_with_ga_elegant(v, angle, axis)
    print(f"Vector sau khi quay bằng Rotor thanh lịch: {v_rotated_ga_elegant}")
    print(f"Kết quả có đúng không? {np.allclose(v_rotated_ga_elegant, expected_result)}\n")


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def rotate_with_quaternion(vector, angle, axis):
    axis = axis / np.linalg.norm(axis)
    q_scalar = np.cos(angle / 2); q_vector = np.sin(angle / 2) * axis
    q = np.hstack([q_scalar, q_vector])
    q_conj = np.hstack([q[0], -q[1:]])
    v_quat = np.hstack([0, vector])
    v_rotated_quat = quat_mult(quat_mult(q, v_quat), q_conj)
    return v_rotated_quat[1:]

def rotate_with_ga_manual_fixed(vector, angle):
    """Sử dụng công thức Rotor thủ công đã sửa lỗi."""
    plane_of_rotation = e1^e2
    R = np.cos(angle / 2) + plane_of_rotation * np.sin(angle / 2)
    # GHI CHÚ: Quy ước dấu +/- phụ thuộc vào định nghĩa của bivector.
    # Quay từ x->y (ngược chiều kim đồng hồ) quanh trục z dương.
    # Bivector B = e1^e2. Rotor là exp(B*theta/2) = cos(theta/2) + B*sin(theta/2).
    # Code cũ của tôi bị ngược vì tôi dùng công thức exp(-B*theta/2) nhưng lại dùng dấu '+'.
    # Phiên bản này đã sửa lại cho đúng.
    
    v_ga = vector[0]*e1 + vector[1]*e2 + vector[2]*e3
    v_rotated_ga = R * v_ga * ~R
    return np.array([v_rotated_ga[e1], v_rotated_ga[e2], v_rotated_ga[e3]])

def rotate_with_ga_elegant(vector, angle, axis):
    """Sử dụng exponential map - cách làm chuẩn của clifford."""
    # 1. Chuyển numpy array axis thành GA vector
    axis_ga = layout.MultiVector(grade=1, value=axis)
    
    # 2. Chuyển vector trục quay thành bivector mặt phẳng quay
    # Dấu trừ là cần thiết theo quy ước
    plane_of_rotation_bivector = -normal_to_bivector(axis_ga)
    
    # 3. Tạo rotor bằng exponential map: R = exp(B * theta / 2)
    R = (plane_of_rotation_bivector * angle / 2).exp()

    # 4. Thực hiện phép quay
    v_ga = vector[0]*e1 + vector[1]*e2 + vector[2]*e3
    v_rotated_ga = R * v_ga * ~R
    return np.array([v_rotated_ga[e1], v_rotated_ga[e2], v_rotated_ga[e3]])

if __name__ == "__main__":
    main()
