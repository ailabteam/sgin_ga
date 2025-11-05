# ex1_benchmark_setup.py

import numpy as np
import time
from clifford.g3 import e1, e2, e3, layout

def main():
    """
    Hàm chính để so sánh phép quay bằng Quaternion và Geometric Algebra.
    Phiên bản cuối cùng, sử dụng các phép toán cơ bản nhất để đảm bảo hoạt động.
    """
    print("--- So sánh phép quay 3D (Phiên bản ổn định) ---")
    
    # Dữ liệu đầu vào
    v = np.array([1.0, 0.0, 0.0])
    angle = np.pi / 2
    axis = np.array([0.0, 0.0, 1.0])
    expected_result = np.array([0.0, 1.0, 0.0])
    
    print(f"Vector ban đầu v: {v}")
    print(f"Góc quay: {np.rad2deg(angle)} độ quanh trục {axis}\n")

    # --- 1. Quaternion (Đã đúng và ổn định) ---
    print("--- 1. Sử dụng Quaternion ---")
    v_rotated_q = rotate_with_quaternion(v, angle, axis)
    print(f"Vector sau khi quay bằng Quaternion: {v_rotated_q}")
    print(f"Kết quả có đúng không? {np.allclose(v_rotated_q, expected_result)}\n")
    
    # --- 2. Geometric Algebra (Sử dụng phép toán đối ngẫu - Dual) ---
    # SỬA LỖI: Thay thế hoàn toàn các hàm tiện ích bị lỗi bằng phương pháp toán học cơ bản.
    print("--- 2. Sử dụng GA (Rotor với phép toán đối ngẫu) ---")
    v_rotated_ga = rotate_with_ga_stable(v, angle, axis)
    print(f"Vector sau khi quay bằng Rotor: {v_rotated_ga}")
    print(f"Kết quả có đúng không? {np.allclose(v_rotated_ga, expected_result)}\n")


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

def rotate_with_ga_stable(vector, angle, axis):
    """
    Thực hiện phép quay bằng GA Rotor, sử dụng các phép toán cơ bản và ổn định nhất.
    """
    # 1. Định nghĩa Pseudoscalar của không gian G3
    I3 = e1^e2^e3
    
    # 2. Chuyển vector trục quay (numpy) thành vector GA
    axis_ga = axis[0]*e1 + axis[1]*e2 + axis[2]*e3
    
    # 3. Tìm bivector mặt phẳng quay bằng cách lấy đối ngẫu (dual) của vector trục quay
    # B = -a*I3
    plane_of_rotation = -axis_ga * I3
    
    # 4. Tạo Rotor bằng công thức Euler: R = exp(B * theta / 2)
    # R = cos(theta/2) + B_normalized * sin(theta/2)
    # Chuẩn hóa bivector để có độ lớn 1
    B_normalized = plane_of_rotation.normal()
    R = np.cos(angle / 2) + B_normalized * np.sin(angle / 2)

    # 5. Thực hiện phép quay
    v_ga = vector[0]*e1 + vector[1]*e2 + vector[2]*e3
    v_rotated_ga = R * v_ga * ~R
    
    # 6. Trả về kết quả dưới dạng numpy array
    return np.array([v_rotated_ga[e1], v_rotated_ga[e2], v_rotated_ga[e3]])

if __name__ == "__main__":
    main()
